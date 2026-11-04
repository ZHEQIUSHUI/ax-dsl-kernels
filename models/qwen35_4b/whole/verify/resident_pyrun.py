"""Seed PYRUN output buffers from RESIDENT, matching the board's persistent in/out buffers.

The simulator's PYRUN path builds ground truth only from the kernel's input
arguments; output arguments start undefined even when the host binds the same
buffer as input and output.  q35_step already monkeypatches the CRUN backend
path.  This extends the same behaviour to PYRUN so the MTP paged KV and the
decode group's recurrent state can be carried across calls in checks.
"""
from __future__ import annotations

import axnn.dsl.builder_ as B
from axnn.dsl import util

RESIDENT = {}
_orig = B.KernelBuilder.init_pyrun_inputs


def _init_pyrun_inputs(self):
    _orig(self)
    if not RESIDENT:
        return
    for chip_ctxs in B.ctx.sub_ctxs:
        ctx_chip = chip_ctxs[-1]
        for name, value in RESIDENT.items():
            if name in ctx_chip.outputs:
                ctx_chip.all_gt[name] = util.torch2np(value.clone(), packed=False)


B.KernelBuilder.init_pyrun_inputs = _init_pyrun_inputs


def set_(**tensors):
    RESIDENT.clear()
    RESIDENT.update(tensors)


def clear():
    RESIDENT.clear()
