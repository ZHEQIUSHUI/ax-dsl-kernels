"""Host scheduling of independent 256-token attention kernels."""

from sgl_kernel_axera.attention.flash_attn import flash_attention
from sgl_kernel_axera.attention.gdn import GDNKernel


def attention_chunks(workspace, metadata):
    chunked = metadata["kernel_tokens"] > 256
    page = metadata["page"]
    rows = min(256, metadata["kernel_tokens"])
    chunks = []
    for index, start in enumerate(range(0, metadata["kernel_tokens"], rows)):
        end = start + rows
        valid_tokens = metadata["valid_tokens"][index] if chunked else metadata["valid_tokens"]
        chunks.append(
            (
                workspace.hidden[start:end].view(1, rows, -1),
                workspace.attention[start:end].view(1, rows, -1),
                valid_tokens,
                metadata["positions"][index] if chunked else metadata["positions"],
                page["indptr"][index] if chunked else page["indptr"],
                page["last_page_len"][index] if chunked else page["last_page_len"],
                metadata["cache_location"][index] if chunked else metadata["cache_location"],
            )
        )
    return chunks


def run_attention(kv_buffer, weights, chunks, page_indices, mrope_section):
    for hidden, output, valid_tokens, positions, indptr, last_len, location in chunks:
        if kv_buffer is None:
            GDNKernel[4, 4](
                hidden,
                None,
                output=None,
                residual_out=output,
                valid_tokens=valid_tokens,
                **weights,
            )
        else:
            flash_attention(
                hidden,
                weights,
                positions,
                output,
                kv_buffer,
                indptr,
                page_indices,
                last_len,
                cache_location=location,
                core_num=4,
                attn_output_gate=True,
                partial_rotary_factor=weights["partial_rotary_factor"],
                has_mrope=positions.ndim == 2,
                mrope_section=mrope_section,
                mrope_interleaved=True,
            )
