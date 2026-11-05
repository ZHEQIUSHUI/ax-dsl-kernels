"""Single-request metadata kernels; runtime values remain on the device."""

import os

import numpy as np
import axnn.dsl as ax

# Shapes below are logical DSL shapes; [1] denotes a tensor, not a Python scalar.
# Kernel tensor arguments are DDR; constants and computed intermediates are OCM unless stated otherwise.
# load/from_numpy return OCM tensors. Without r=, arithmetic/gather/concat/cast also return OCM tensors.
# store's arrow describes its data write; its Python return is None. select_scatter updates and returns r.
# Slices/reshapes are storage views and preserve dtype; [*A,N] means [N] or [3,N] below.
# Python arithmetic on DSL tensors records add/sub/mul with the shapes and dtypes annotated below.
# Slot storage uses little-endian uint16 pairs for the low 32 bits of each slot.


def divide_by_page_size(value):
    # Input value[N]: int32 or uint16; output [N]: int32. N may be 1 or a vector length.
    # Power-of-two scaling is exact; AxDiv's reciprocal approximation can
    # round an exact page boundary down before integer truncation.
    # from_numpy: np[1] float32 -> scale[1] fp32.
    scale = ax.dsl.from_numpy(np.array([1 / 256], dtype=np.float32))
    # mul: value[N] int32/uint16, scale[1] fp32 -> [N] fp32.
    # floor: mul result[N] fp32 -> returned[N] int32 OCM; the result is a tensor even when N=1.
    return ax.dsl.floor(ax.dsl.mul(value, scale, output_dtype=ax.fp32))


def page_and_offset(words):
    """Divide an int32 slot by 256 without rounding its high address bits.

    Integer AxDiv uses floating-point arithmetic. Splitting a little-endian
    slot into uint16 words keeps every intermediate exactly representable.
    """
    # Input words[2]: uint16; returned page[1] and offset[1]: int32.
    # cast: words[2] uint16 -> pair[2] int32; both slices are contiguous [1] views.
    pair = ax.dsl.cast(words, dtype=ax.int32, output_rounding=1)
    # Slice views: pair[2] int32 -> low[1] and high[1] int32, sharing pair's OCM allocation.
    low = pair[:1]
    high = pair[1:2]
    # from_numpy: np[1] int32 -> size[1] int32.
    size = ax.dsl.from_numpy(np.array([256], dtype=np.int32))
    # Helper: low[1] int32 -> low_page[1] int32.
    low_page = divide_by_page_size(low)
    # mul: high[1] int32 * size[1] int32 -> [1] int32; add low_page[1] -> page[1] int32.
    page = high * size + low_page
    # mul: low_page[1] int32 * size[1] int32 -> [1] int32; sub from low[1] -> offset[1] int32.
    return page, low - low_page * size


def load_token_ids(words, count):
    # N=count is a Python shape parameter; words[N*S]: int32 DDR, S=1 or 2 for int32/int64 IDs.
    # Only each ID's low int32 word is consumed; the result is [N] int32.
    stride = words.shape[0] // count
    if count == 1 or stride == 1:
        # load: words[:N][N] int32 DDR -> [N] int32 OCM.
        return ax.dsl.load(words[:count])
    # from_numpy: np[N] int32 -> index[N] int32, selecting every S-th word.
    index = ax.dsl.from_numpy(np.arange(count, dtype=np.int32) * stride)
    # index_select(dim=0): words[N*S] int32 DDR, index[N] int32 OCM -> [N] int32 OCM.
    return ax.dsl.index_select(words, index, dim=0)


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def release_page_ids(slot_words, parameters, page_bitmap):
    """Mark distinct pool pages from arbitrary slot IDs, retaining a free-group snapshot.

    Parameters are [word_start, words_per_slot, slot_count, 0, 0, 0, 0, 0].
    Slot IDs are valid int32/int64 pool indices; only their low 32 bits are needed.
    The bitmap includes reserved page zero and remains an input-only mutable state.
    """
    # Inputs: slot_words[W] uint16, parameters[8] int32; state page_bitmap[P+1] uint8.
    # load: parameters[8] int32 DDR -> params[8] int32 OCM.
    params = ax.dsl.load(parameters)
    # load: page_bitmap[P+1] uint8 DDR -> marked[P+1] uint8 OCM.
    marked = ax.dsl.load(page_bitmap)
    # from_numpy: scalar zero/one/256 -> [1] int32 constants; lanes[256] int32.
    zero = ax.dsl.from_numpy(np.array([0], dtype=np.int32))
    one = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    size = ax.dsl.from_numpy(np.array([256], dtype=np.int32))
    lanes = ax.dsl.from_numpy(np.arange(256, dtype=np.int32))
    # from_numpy: updates[256] uint8; repeated page indices all write the same one.
    updates = ax.dsl.from_numpy(np.ones(256, dtype=np.uint8))
    # add/divide_by_page_size: ceil(slot_count[1]/256) -> tile_count[1] int32.
    tile_count = divide_by_page_size(params[2:3] + size - one)

    def mark_tile(tile, bitmap):
        # mul/add/minimum: tile[1], lanes[256], count[1] -> valid logical[256] int32.
        # The tail repeats the last valid slot, so it cannot mark an unrelated page.
        logical = ax.dsl.minimum(tile * size + lanes, params[2:3] - one)
        # mul/add: word_start[1] + logical[256] * stride[1] -> indices[256] int32.
        indices = params[:1] + logical * params[1:2]
        # index_select: slot_words[W] uint16 -> low[256] and high[256] uint16 OCM.
        low = ax.dsl.index_select(slot_words, indices, dim=0)
        high = ax.dsl.index_select(slot_words, indices + one, dim=0)
        # mul/add: high * 256 + floor(low/256) -> pages[256] uint32 directly.
        # Each uint16 limb is exact; a complete int32/int64 slot never passes through FP32.
        pages = ax.dsl.add(
            ax.dsl.mul(high, size, output_dtype=ax.int32), divide_by_page_size(low), output_dtype=ax.uint32
        )
        # select_scatter: updates[256] uint8, pages[256] uint32 -> bitmap[P+1] in place.
        ax.dsl.select_scatter(updates, pages, dim=0, r=bitmap)
        return bitmap

    # range: runtime tile_count[1] int32; carry marked[P+1] uint8 across all input tiles.
    (marked,) = ax.dsl.range(zero, zero, tile_count, one, mark_tile, marked)
    # store: final marked[P+1] uint8 OCM -> persistent page_bitmap[P+1] uint8 DDR.
    ax.dsl.store(marked, r=page_bitmap)
    return ()


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def publish_free_pages(page_bitmap, parameters, free_pages, released_count):
    """Compact marked page IDs into the retained pool and reset the free-group bitmap.

    parameters[:2] contain [destination_base, grow_backwards]. The destination is
    base-count for free_storage, or base for append-only release_storage.
    """
    # Inputs/state: page_bitmap[P+1] uint8, parameters[8] int32, free_pages[P] int32.
    # Output: released_count[1] int32. This call always has at least one marked page.
    # load: bitmap[P+1] uint8 and parameters[8] int32 DDR -> OCM values of the same shapes.
    marked = ax.dsl.load(page_bitmap)
    params = ax.dsl.load(parameters)
    # nonzero: marked[P+1] uint8 -> indices[ceil((P+1)/32)*256] uint32, count[1] uint32.
    # Only the first count indices are valid; no dynamic output allocation is needed.
    indices, count = ax.dsl.nonzero(marked)
    # cast: count[1] uint32 -> count_i32[1] int32 (bounded by the configured pool capacity).
    count_i32 = ax.dsl.cast(count, dtype=ax.int32)
    # from_numpy: one[1] int32; minimum: arange(P)[P], count[1]-one[1] -> logical[P] int32.
    one = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    logical = ax.dsl.minimum(ax.dsl.from_numpy(np.arange(free_pages.shape[0], dtype=np.int32)), count_i32 - one)
    # index_select: indices[aligned] uint32, logical[P] int32 -> pages[P] uint32.
    pages = ax.dsl.index_select(indices, logical, dim=0)
    # mul/sub/add: base[1] - count[1]*grow_backwards[1] + logical[P] -> destinations[P] uint32.
    destinations = ax.dsl.add(params[:1] - count_i32 * params[1:2], logical, output_dtype=ax.uint32)
    # select_scatter: pages[P] uint32, destinations[P] uint32 -> free_pages[P] int32 DDR in place.
    # Repeated tail lanes write the same final valid page to the same destination.
    ax.dsl.select_scatter(pages, destinations, dim=0, r=free_pages)
    # store: count_i32[1] int32 OCM -> released_count[1] int32 DDR.
    ax.dsl.store(count_i32, r=released_count)
    # zeros/store: reset[P+1] uint8 OCM -> page_bitmap[P+1] uint8 DDR for the next group.
    ax.dsl.store(ax.dsl.zeros(page_bitmap.shape, dtype=ax.uint8), r=page_bitmap)
    return released_count


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def merge_page_ids(pages, parameters, free_pages):
    """Append pending IDs without moving existing IDs in the fixed page pool."""
    # P=pages.shape[0], F=free_pages.shape[0].
    # Inputs: pages[P] int32, parameters[8] int32; state free_pages[F] int32.
    # load: parameters[8] int32 DDR -> params[8] int32 OCM.
    params = ax.dsl.load(parameters)
    # from_numpy: np[1] int32 -> one[1] int32.
    one = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    # from_numpy: arange(P)[P] int32 -> [P] int32 OCM.
    # sub: params[3:4][1] int32 - one[1] int32 -> last valid page index[1] int32.
    # minimum: arange[P] int32, last index[1] int32 -> logical[P] int32.
    logical = ax.dsl.minimum(ax.dsl.from_numpy(np.arange(pages.shape[0], dtype=np.int32)), params[3:4] - one)
    # index_select(dim=0): pages[P] int32 DDR, logical[P] int32 -> values[P] int32 OCM.
    values = ax.dsl.index_select(pages, logical, dim=0)
    # add: params[5:6][1], logical[P] int32 -> destination[P] uint32 directly.
    destination = ax.dsl.add(params[5:6], logical, output_dtype=ax.uint32)
    # select_scatter(dim=0): values[P] int32, destination[P] uint32 -> free_pages[F] int32 DDR in place.
    ax.dsl.select_scatter(values, destination, dim=0, r=free_pages)
    return ()


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def prepare_positions(length_words, positions, start_loc, decode: bool):
    # N=positions.shape[0]; length_words[1 or 2], positions[N], optional start_loc[1]: int32.
    # load: length_words[:1][1] int32 DDR -> start[1] int32 OCM.
    start = ax.dsl.load(length_words[:1])
    if decode:
        # First from_numpy: one np[1] int32 -> [1] int32 OCM.
        # sub: start[1] int32 - one[1] int32 -> candidate position[1] int32.
        # Second from_numpy: zero np[1] int32 -> [1] int32 OCM.
        # maximum: candidate[1] int32, zero[1] int32 -> start[1] int32, including an empty sequence.
        start = ax.dsl.maximum(
            start - ax.dsl.from_numpy(np.array([1], dtype=np.int32)), ax.dsl.from_numpy(np.array([0], dtype=np.int32))
        )
    if positions.shape[0] == 1:
        # store: start[1] int32 OCM -> positions[1] int32 DDR; no addition of zero.
        ax.dsl.store(start, r=positions)
    else:
        # from_numpy: arange(N)[N] int32 -> [N] int32 OCM.
        # add: start[1] int32, arange[N] int32 -> [N] int32 by scalar broadcasting.
        # store: resulting [N] int32 OCM -> positions[N] int32 DDR.
        ax.dsl.store(start + ax.dsl.from_numpy(np.arange(positions.shape[0], dtype=np.int32)), r=positions)
    if start_loc is not None:
        # zeros: shape (1,), dtype=int32 -> [1] int32 OCM.
        # store: zeros[1] int32 OCM -> start_loc[1] int32 DDR; returns None.
        ax.dsl.store(ax.dsl.zeros((1,), dtype=ax.int32), r=start_loc)
    return positions, start_loc


def slot_words(page, offsets):
    """Encode slots as two uint16 words, preserving all int32 address bits."""
    # N=offsets.shape[0]; page[1 or N], offsets[N]: int32; result [N,2]: uint16.
    # from_numpy: np[1] int32 -> size[1] int32.
    size = ax.dsl.from_numpy(np.array([256], dtype=np.int32))
    # Helper: page[1 or N] int32 -> high[1 or N] int32.
    high = divide_by_page_size(page)
    # mul/sub: high[1 or N] int32 * size[1] int32, then page[1 or N] minus that -> [1 or N] int32.
    # mul/add: the result * size[1], then + offsets[N] -> low[N] int32 by broadcasting.
    low = (page - high * size) * size + offsets
    # First reshape: low[N] int32 -> [N,1] int32 view.
    # expand: high[1 or N] int32, shape=low.shape -> [N] int32 OCM.
    # Second reshape: expanded high[N] int32 -> [N,1] int32 view.
    # concat(dim=1): low[N,1], expanded high[N,1] int32 -> [N,2] int32.
    # cast: [N,2] int32 -> [N,2] uint16, in low-word/high-word order.
    return ax.dsl.cast(
        ax.dsl.concat(low.reshape(-1, 1), ax.dsl.expand(high, shape=low.shape).reshape(-1, 1), dim=1),
        dtype=ax.uint16,
        output_rounding=1,
    )


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def allocate_extend(free_page_words, prefix_len, seq_len, output_words, last_loc_words):
    """Allocate one partial page or several complete pages from runtime page IDs.

    Multi-page callers supply page-aligned prefixes, full output pages, and
    None for the three unused length/previous-slot inputs.
    """
    if free_page_words.shape[0] > 1:
        # Aligned multi-page prefill: pages[P] int32 DDR -> [P,1] int32 OCM.
        pages = ax.dsl.load(free_page_words).reshape(-1, 1)
        # floor/mul/sub: pages[P,1] -> high/low[P,1] int32, preserving full slot addresses.
        high = divide_by_page_size(pages)
        size = ax.dsl.from_numpy(np.array([256], dtype=np.int32))
        low = (pages - high * size) * size
        # add/expand: page words[P,1] -> [P,256] int32 in logical token order.
        low = low + ax.dsl.from_numpy(np.arange(256, dtype=np.int32)).reshape(1, 256)
        high = ax.dsl.expand(high, shape=low.shape)
        # concat/cast/store: two [P,256,1] columns -> [P*512] uint16 DDR.
        words = ax.dsl.cast(
            ax.dsl.concat(low.reshape(-1, 1), high.reshape(-1, 1), dim=1), dtype=ax.uint16, output_rounding=1
        )
        ax.dsl.store(words.reshape(-1), r=output_words)
        return output_words
    # free_page_words/prefix_len/seq_len[1]: int32 DDR;
    # last_loc_words[2]: uint16 DDR (low 32 bits of the previous slot);
    # output_words[512]: uint16 DDR, caller-owned full allocation block.
    # from_numpy: np[1] int32 -> zero[1] int32.
    zero = ax.dsl.from_numpy(np.array([0], dtype=np.int32))
    # from_numpy: np[1] int32 -> one[1] int32.
    one = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    # from_numpy: np[1] int32 -> size[1] int32.
    size = ax.dsl.from_numpy(np.array([256], dtype=np.int32))
    # load: prefix_len[1] int32 DDR -> prefix[1] int32 OCM.
    prefix = ax.dsl.load(prefix_len)
    # load: seq_len[1] int32 DDR -> [1] int32 OCM.
    # sub: loaded sequence length[1] int32 - prefix[1] int32 -> count[1] int32.
    count = ax.dsl.load(seq_len) - prefix
    # divide_by_page_size: prefix[1] int32 -> page number[1] int32 OCM.
    # mul/sub: page number[1] * size[1], then subtract from prefix[1] -> offset[1] int32.
    offset = prefix - divide_by_page_size(prefix) * size
    # Inner minimum: offset[1] int32, one[1] int32 -> partial-page flag[1] int32 (0/1).
    # sub/mul: (size[1]-offset[1]) * flag[1] -> remaining capacity[1] int32, zero at page boundaries.
    # Outer minimum: count[1] int32, remaining capacity[1] int32 -> existing[1] int32.
    existing = ax.dsl.minimum(count, (size - offset) * ax.dsl.minimum(offset, one))
    # load: last_loc_words[2] uint16 DDR -> previous[2] uint16 OCM.
    # The framework provides -1 for an empty prefix; use_old then masks it out.
    previous = ax.dsl.load(last_loc_words)
    # load: free_page_words[1] int32 DDR -> page[1] int32 OCM.
    page = ax.dsl.load(free_page_words)
    # divide_by_page_size: page[1] int32 -> high[1] int32 OCM.
    # Decode the new page once; retain the old slot's two words without a round trip.
    high = divide_by_page_size(page)
    # mul/sub/mul: page[1] int32 - high[1]*size[1], then * size[1] -> low[1] int32.
    low = (page - high * size) * size
    # from_numpy: np[256] int32 -> logical[256] int32.
    logical = ax.dsl.from_numpy(np.arange(256, dtype=np.int32))
    # sub: existing[1] int32 - logical[256] int32 -> [256] int32 by scalar broadcasting.
    # maximum: difference[256] int32, zero[1] int32 -> nonnegative[256] int32.
    # minimum: nonnegative[256] int32, one[1] int32 -> use_old[256] int32 (0/1).
    use_old = ax.dsl.minimum(ax.dsl.maximum(existing - logical, zero), one)
    # Inner add(output_dtype=int32): previous[:1][1] uint16, one[1] int32 -> previous low word + 1[1] int32.
    # First add/mul: that [1] int32 + logical[256], then * use_old[256] -> retained-page term[256] int32.
    # Second add/sub/mul: (low[1] + logical[256] - existing[1]) * (one[1]-use_old[256]) -> new-page term[256] int32.
    # Outer add(output_dtype=uint16): both [256] int32 terms -> low[256] uint16 directly.
    low = ax.dsl.add(
        (ax.dsl.add(previous[:1], one, output_dtype=ax.int32) + logical) * use_old,
        (low + logical - existing) * (one - use_old),
        output_dtype=ax.uint16,
    )
    # Inner mul(output_dtype=int32): previous[1:2][1] uint16, use_old[256] int32 -> retained high[256] int32.
    # sub/mul: high[1] int32 * (one[1] int32 - use_old[256] int32) -> new high[256] int32.
    # Outer add(output_dtype=uint16): retained/new high[256] int32 -> high[256] uint16 directly.
    high = ax.dsl.add(
        ax.dsl.mul(previous[1:2], use_old, output_dtype=ax.int32), high * (one - use_old), output_dtype=ax.uint16
    )
    # Input reshapes: low/high[256] uint16 -> two [256,1] uint16 views.
    # concat(dim=1): the two [256,1] uint16 columns -> interleaved slots[256,2] uint16 OCM.
    # Output reshape: [256,2] uint16 -> words[512] uint16 view, still in low-word/high-word order.
    words = ax.dsl.concat(low.reshape(-1, 1), high.reshape(-1, 1), dim=1).reshape(-1)
    # store: words[512] uint16 OCM -> output_words[512] uint16 DDR.
    ax.dsl.store(words, r=output_words)
    return output_words


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def allocate_decode(
    last_loc_words, free_page_words, seq_len, output_words, seq_len_out=None, orig_len=None, orig_len_out=None
):
    """Allocate a slot; optionally advance lengths into round-owned outputs.

    The standalone ABI consumes a post-allocation length. With length outputs,
    seq_len instead holds the old length, shared with the previous forward.
    """
    # DDR last_loc_words/output_words[2]: uint16; free_page_words[1]: int32.
    # Length inputs/outputs[1 or 2]: int32 words of nonnegative int32/int64 lengths.
    one = ax.dsl.full((1,), ax.int32, fill_value=1)
    size = ax.dsl.full((1,), ax.int32, fill_value=256)
    # load: low word of seq_len -> prefix[1] int32 OCM.
    prefix = ax.dsl.load(seq_len[:1])
    if seq_len_out is None:
        prefix = prefix - one
    else:
        # add: old length[1] int32 -> next length[1] int32 OCM.
        next_len = prefix + one
        # concat: zero-extend to int64's two int32 words when requested.
        if seq_len_out.shape[0] == 2:
            next_len = ax.dsl.concat(next_len, ax.dsl.zeros((1,), dtype=ax.int32), dim=0)
        # store: int32[1 or 2] -> caller-owned int32/int64 length buffer.
        ax.dsl.store(next_len, r=seq_len_out)
        # load/add: orig_len[:1] int32 DDR -> next original length[1] int32 OCM.
        next_orig = ax.dsl.load(orig_len[:1]) + one
        if orig_len_out.shape[0] == 2:
            next_orig = ax.dsl.concat(next_orig, ax.dsl.zeros((1,), dtype=ax.int32), dim=0)
        # store: int32[1 or 2] -> caller-owned original-length buffer.
        ax.dsl.store(next_orig, r=orig_len_out)
    # Runtime modulo: prefix[1] -> offset[1] int32 in [0,255].
    offset = prefix - divide_by_page_size(prefix) * size
    # minimum/sub: offset[1] int32 -> new_page[1] int32 flag (0/1).
    new_page = one - ax.dsl.minimum(offset, one)
    # load: last_loc_words[2] uint16 DDR -> previous[2] uint16 OCM.
    previous = ax.dsl.load(last_loc_words)
    # load: free_page_words[1] int32 DDR -> page[1] int32 OCM.
    page = ax.dsl.load(free_page_words)
    # Encode the new page into uint16[2] OCM, without converting the full slot to FP32.
    allocated = slot_words(page, ax.dsl.zeros((1,), dtype=ax.int32)).reshape(2)
    # add: uint16[2] + uint16[2] -> int32[2]. In-page increments cannot carry
    # into the high word; a page boundary selects the new-page value instead.
    continued = ax.dsl.add(previous, ax.dsl.from_numpy(np.array([1, 0], dtype=np.uint16)), output_dtype=ax.int32)
    # mul/add: two int32[2] alternatives -> uint16[2] directly.
    words = ax.dsl.add(
        ax.dsl.mul(allocated, new_page, output_dtype=ax.int32), continued * (one - new_page), output_dtype=ax.uint16
    )
    # store: OCM uint16[2] -> DDR output_words[2], optionally bound to a request-table slot.
    ax.dsl.store(words, r=output_words)
    return output_words, seq_len_out, orig_len_out


@ax.builder(run_mode=ax.RunMode(os.getenv("RUN_MODE", "CRUN")))
def prepare_attention_metadata(
    request_table_words,
    request_index_words,
    input_ids,
    positions,
    length_words,
    token_ids_out,
    positions_out,
    indices_out,
    indptr_out,
    last_page_len_out,
    location_out,
    decode: bool,
    generated_positions_out=None,
    start_loc_out=None,
    valid_tokens_out=None,
    token_count=None,
):
    # With token_count, IDs/positions cover the physical bucket. The H2D
    # staging pads ID storage with zero; masking IDs through FP32 would round
    # valid int32 values above 2**24. Only positions require a runtime mask.
    # V=valid tokens; Q=1 for decode or ceil(V/256)*256 for prefill.
    # Multi-block prefill stores positions as [B,256]/[B,3,256] and the
    # indptr/last_page_len/location outputs as [B,2]/[B,32]/[B,2], B=Q/256.
    # P=indices_out.shape[0], R=request_table_words.shape[0], C=request_table_words.shape[1].
    # A=() for ordinary RoPE, or (3,) for MRoPE; positions have shape [*A,V].
    # Inputs: request_table_words[R,C,2] uint16, request_index_words[1 or 2] int32,
    # length_words[1 or 2] int32. The request table includes the newly allocated pages.
    # Optional inputs: input_ids[V*S] int32 (S=1/2 for int32/int64 IDs), positions[*A,V] int32,
    # Outputs: positions_out[*A,Q], indices_out[P], indptr_out[2], last_page_len_out[32], location_out[2].
    # Optional outputs: token_ids_out[Q], generated_positions_out[*A,V], start_loc_out[1].
    # All outputs are int32 DDR tensors. Only decode selects a recorded branch.
    chunked = len(location_out.shape) == 2
    chunks = location_out.shape[0] if chunked else 1
    row_capacity = request_table_words.shape[1]
    # FP32 represents all page-aligned indices in the uint32 address range exactly.
    assert row_capacity % 256 == 0
    assert request_table_words.shape[0] * row_capacity <= 2**32
    # load: low word of the runtime int32/int64 request ID -> request[1] int32 OCM.
    request = ax.dsl.load(request_index_words[:1])
    # load: length_words[:1][1] int32 DDR -> prefix[1] int32 OCM.
    prefix = ax.dsl.load(length_words[:1])
    # from_numpy: np[1] int32 -> size[1] int32.
    size = ax.dsl.from_numpy(np.array([256], dtype=np.int32))
    # from_numpy: np[1] int32 -> zero[1] int32.
    zero = ax.dsl.from_numpy(np.array([0], dtype=np.int32))
    # from_numpy: np[1] int32 -> one[1] int32.
    one = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
    if decode:
        # sub: prefix[1], one[1] int32 -> prefix[1] int32.
        prefix = prefix - one
    # Multi-block outputs retain one [2]/[32]/[2] metadata row per 256-token call.
    # add: runtime prefix[1] + block offsets[B] -> block_prefix[B] int32 OCM.
    block_offsets = zero
    if chunked:
        block_offsets = ax.dsl.from_numpy(np.arange(chunks, dtype=np.int32) * 256)
    block_prefix = prefix + block_offsets if chunked else prefix
    # add/sub: prefix[1] int32 + size[1] int32 - one[1] int32 -> rounded-up numerator[1] int32.
    # divide_by_page_size: numerator[1] int32 -> page_count[1] int32 OCM.
    page_count = divide_by_page_size(block_prefix + size - one)
    logical_pages = np.arange(indices_out.shape[0], dtype=np.int32)
    # from_numpy: np[P] int32 -> logical[P] int32.
    logical = ax.dsl.from_numpy(logical_pages)
    # from_numpy: row capacity np[1] int32 -> [1] int32 OCM.
    # mul(output_dtype=uint32): request[1] * capacity[1] -> row_base[1] uint32 OCM.
    row_base = ax.dsl.mul(request, ax.dsl.from_numpy(np.array([row_capacity], dtype=np.int32)), output_dtype=ax.uint32)
    # from_numpy: page-start token indices[P] int32 -> [P] int32 OCM.
    # add(output_dtype=uint32): row_base[1] + clamped page starts[P] -> slot_indices[P] uint32 OCM.
    slot_indices = ax.dsl.add(
        row_base,
        ax.dsl.from_numpy(np.minimum(logical_pages, row_capacity // 256 - 1) * 256),
        output_dtype=ax.uint32,
    )
    # reshape: table[R,C,2] -> [R*C,2] uint16 DDR view; no data movement.
    # index_select(dim=0): table view and slot_indices[P] -> selected[P,2] uint16 OCM.
    # cast(dtype=int32): selected[P,2] -> pair[P,2] int32 OCM, preserving both slot words exactly.
    pair = ax.dsl.cast(ax.dsl.index_select(request_table_words.reshape(-1, 2), slot_indices, dim=0), dtype=ax.int32)
    # Slice views: pair[P,2] -> low[P,1] and high[P,1] int32 OCM.
    low, high = pair[:, :1], pair[:, 1:2]
    # mul/add/divide_by_page_size: high[P,1]*256 + floor(low[P,1]/256) -> pages[P,1] int32 OCM.
    pages = high * size + divide_by_page_size(low)
    # minimum: nonnegative request[1] int32 -> real_request[1] int32 mask (0 for the dummy row).
    real_request = ax.dsl.minimum(request, one)
    # reshape: logical[P] -> [P,1] int32 OCM view.
    # mul/add: select real pages or dummy identity pages at runtime; result [P,1] int32 OCM.
    # reshape: [P,1] -> pages[P] int32 OCM view.
    pages = (pages * real_request + logical.reshape(-1, 1) * (one - real_request)).reshape(-1)
    # mul: nonnegative block_prefix[B] int32 / 256 -> [B] FP32 OCM.
    # cast(toward-zero): [B] FP32 -> logical page indices[B] int32 OCM.
    current = ax.dsl.cast(
        ax.dsl.mul(block_prefix, ax.dsl.full((1,), ax.fp32, fill_value=1 / 256), output_dtype=ax.fp32),
        dtype=ax.int32,
        output_rounding=1,
    )
    # index_select: physical pages[P] int32 OCM -> current pages[B] int32 OCM.
    # Read before masking history: a newly allocated page is not history yet.
    page = ax.dsl.index_select(pages, current, dim=0)
    # sub/mul: prefix[B] - logical_page[B]*256 -> within-page offsets[B] int32 OCM.
    offset = block_prefix - current * size
    # sub: page_count[1] int32 - logical[P] int32 -> remaining[P] int32 by scalar broadcasting.
    # maximum: remaining[P] int32, zero[1] int32 -> nonnegative[P] int32.
    # minimum: nonnegative[P] int32, one[1] int32 -> keep[P] int32 (0/1).
    history_pages = page_count[chunks - 1 : chunks] if chunked else page_count
    keep = ax.dsl.minimum(ax.dsl.maximum(history_pages - logical, zero), one)
    # mul: pages[P], keep[P] int32 -> pages[P] int32; unused page entries become zero.
    pages = pages * keep
    # store: pages[P] int32 OCM -> indices_out[P] int32 DDR.
    ax.dsl.store(pages, r=indices_out)
    # concat(dim=0): zero[1] int32, page_count[1] int32 -> [2] int32 OCM.
    # store: concat result[2] int32 OCM -> indptr_out[2] int32 DDR; returns None.
    if chunked:
        # concat: zero[B,1], page_count[B,1] int32 -> indptr[B,2] int32 DDR.
        indptr = ax.dsl.concat(ax.dsl.zeros((chunks, 1), dtype=ax.int32), page_count.reshape(chunks, 1), dim=1)
    else:
        indptr = ax.dsl.concat(zero, page_count, dim=0)
    ax.dsl.store(indptr, r=indptr_out)
    # sub/mul/sub: prefix[1] - (page_count[1]-one[1])*size[1] -> tail candidate[1] int32.
    # maximum: tail candidate[1] int32, zero[1] int32 -> last_len[1] int32.
    last_len = ax.dsl.maximum(block_prefix - (page_count - one) * size, zero)
    # Inner minimum: prefix[1] int32, one[1] int32 -> has-prefix flag[1] int32 (0/1).
    # mul: flag[1] int32 * size[1] int32 -> limit[1] int32 (0 or 256).
    # Outer minimum: last_len[1] int32, limit[1] int32 -> last_len[1] int32, zero for an empty prefix.
    last_len = ax.dsl.minimum(last_len, ax.dsl.minimum(block_prefix, one) * size)
    # zeros: shape (31,), dtype=int32 -> [31] int32 OCM.
    # concat(dim=0): last_len[1] int32, zeros[31] int32 -> padded lengths[32] int32 OCM.
    # store: padded lengths[32] int32 OCM -> last_page_len_out[32] int32 DDR; returns None.
    if chunked:
        # concat: lengths[B,1] + zeros[B,31] -> padded lengths[B,32] int32 DDR.
        last_len = ax.dsl.concat(last_len.reshape(chunks, 1), ax.dsl.zeros((chunks, 31), dtype=ax.int32), dim=1)
    else:
        last_len = ax.dsl.concat(last_len, ax.dsl.zeros((31,), dtype=ax.int32), dim=0)
    ax.dsl.store(last_len, r=last_page_len_out)

    generated = generated_positions_out is not None
    mrope = (generated_positions_out if generated else positions).ndim == 2
    if generated:
        count = generated_positions_out.shape[-1]
        # from_numpy: arange(V)[V] int32 -> [V] int32 OCM.
        # add: prefix[1] int32, arange[V] int32 -> pos[V] int32 by scalar broadcasting.
        token_rows = ax.dsl.from_numpy(np.arange(count, dtype=np.int32))
        pos = prefix + token_rows
        generated_pos = ax.dsl.expand(pos.reshape(1, count), shape=generated_positions_out.shape) if mrope else pos
        # store: pos[*A,V] int32 OCM -> generated_positions_out[*A,V] int32 DDR.
        ax.dsl.store(generated_pos, r=generated_positions_out)
    else:
        count = positions.shape[-1]
        # load: positions[*A,V] int32 DDR -> pos[*A,V] int32 OCM.
        pos = ax.dsl.load(positions)
    if start_loc_out is not None:
        # store: zero[1] int32 OCM -> start_loc_out[1] int32 DDR.
        ax.dsl.store(zero, r=start_loc_out)
    rows = chunks * 256 if chunked else positions_out.shape[-1]
    tail = rows - count
    live_count = ax.dsl.load(token_count) if token_count is not None else None
    if valid_tokens_out is not None:
        # One fixed-size runtime length per attention chunk; the consumer's
        # kernel signature does not depend on the number of real tail tokens.
        total = live_count if live_count is not None else ax.dsl.full((1,), ax.int32, count)
        valid = ax.dsl.minimum(total - block_offsets, size)
        ax.dsl.store(valid.reshape(*valid_tokens_out.shape), r=valid_tokens_out)
    if live_count is not None:
        # Physical inputs cover a fixed bucket; only the length value varies.
        if not generated:
            token_rows = ax.dsl.from_numpy(np.arange(count, dtype=np.int32))
        keep = ax.dsl.minimum(ax.dsl.maximum(live_count - token_rows, zero), one)
        pos = ax.dsl.mul(pos, keep, output_dtype=ax.int32)
    if tail:
        # zeros: shape (*A,Q-V), dtype=int32 -> [*A,Q-V] int32 OCM.
        # concat(dim=-1): pos[*A,V] int32, zeros[*A,Q-V] int32 -> pos[*A,Q] int32 OCM.
        pos = ax.dsl.concat(pos, ax.dsl.zeros((*pos.shape[:-1], tail), dtype=ax.int32), dim=-1)
    if token_ids_out is not None:
        # load_token_ids: input_ids[V*S] int32 DDR, Python count=V -> ids[V] int32 OCM.
        ids = load_token_ids(input_ids, count)
        if tail:
            # zeros: shape (Q-V,), dtype=int32 -> [Q-V] int32 OCM.
            # concat(dim=0): ids[V] int32, zeros[Q-V] int32 -> ids[Q] int32 OCM.
            ids = ax.dsl.concat(ids, ax.dsl.zeros((tail,), dtype=ax.int32), dim=0)
        # store: ids[Q] int32 OCM -> token_ids_out[Q] int32 DDR.
        ax.dsl.store(ids, r=token_ids_out)
    # store: pos[*A,Q] int32 OCM -> positions_out[*A,Q] int32 DDR.
    if generated and mrope:
        # Text axes are identical. Mask once, then expand into the final layout
        # instead of transposing a materialized [3,B,256] position tensor.
        shape = (chunks, 1, 256) if chunked else (1, rows)
        pos = ax.dsl.expand(pos.reshape(*shape), shape=positions_out.shape)
    elif chunked:
        if mrope:
            # transpose: [3,B,256] -> [B,3,256], so every M-RoPE block is contiguous.
            pos = ax.dsl.transpose(pos.reshape(3, chunks, 256), perm=(1, 0, 2))
        else:
            pos = pos.reshape(chunks, 256)
    ax.dsl.store(pos, r=positions_out)
    if chunked:
        # reshape: pages/offsets[B] -> [B,1] int32 OCM views.
        page, offset = page.reshape(chunks, 1), offset.reshape(chunks, 1)
    # concat(dim=0): page[1] int32, offset[1] int32 -> location[2] int32 OCM.
    # store: location[2] int32 OCM -> location_out[2] int32 DDR; returns None.
    ax.dsl.store(ax.dsl.concat(page, offset, dim=1 if chunked else 0), r=location_out)
    return (
        token_ids_out,
        positions_out,
        indices_out,
        indptr_out,
        last_page_len_out,
        location_out,
        generated_positions_out,
        start_loc_out,
        valid_tokens_out,
    )
