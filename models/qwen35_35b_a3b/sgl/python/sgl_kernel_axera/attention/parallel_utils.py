"""Shared launch-grid helpers for AXERA attention kernels."""

import os

import axnn.dsl as ax


SUPPORTED_CORE_NUMS = (1, 4)


def attention_core_num(core_num: int | None = None) -> int:
    """Return the configured attention launch width.

    The DSV4 production layout remains four-stream (the third residual axis is
    always four), but the AX launch grid may use one or four cores.  Keeping
    this choice in one helper prevents individual attention variants from
    silently falling back to a hard-coded four-core launch.
    """
    if core_num is None:
        raw = os.environ.get("SGLANG_AXERA_DSV4_CORE_NUM", "4").strip()
        try:
            core_num = int(raw)
        except ValueError as exc:
            raise ValueError(f"SGLANG_AXERA_DSV4_CORE_NUM must be an integer, got {raw!r}") from exc
    if core_num not in SUPPORTED_CORE_NUMS:
        raise ValueError(f"attention core_num must be one of {SUPPORTED_CORE_NUMS}, got {core_num}")
    return core_num


def validate_core_num(core_num: int, work_items: int, work_name: str) -> None:
    """Validate the AX8860 launch grid for independently owned work items."""
    if core_num not in SUPPORTED_CORE_NUMS:
        raise ValueError(f"core_num must be one of {SUPPORTED_CORE_NUMS}, got {core_num}")
    if work_items < core_num:
        raise ValueError(f"{work_name} ({work_items}) must be no smaller than core_num ({core_num})")


def linear_program_slice(work_items: int) -> slice:
    """Return the current program's non-overlapping contiguous work slice."""
    core_id = ax.program_id(0)
    core_num = ax.num_programs(0)
    assert core_num in SUPPORTED_CORE_NUMS
    assert work_items >= core_num
    # Distribute the remainder across the first programs.  A ceil-divided
    # stride can leave trailing cores with an empty slice (for example five
    # rows over four cores), which is not representable by the DSL reshape
    # operations used by the row kernels.
    base, remainder = divmod(work_items, core_num)
    start = core_id * base + min(core_id, remainder)
    stop = start + base + int(core_id < remainder)
    return slice(start, stop)


__all__ = ["SUPPORTED_CORE_NUMS", "attention_core_num", "linear_program_slice", "validate_core_num"]
