#!/usr/bin/env python3
"""Assemble the dumped real pages into the big-kernel input and reference sets."""
import array
import os
import shutil

R = "/data/qiushui/pf35"


def cat(parts, dst):
    with open(dst, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)


def fresh(dst):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)


def copy_inputs(src_call, dst, skip=()):
    for name in os.listdir(src_call):
        if name.startswith("out_") or name in skip:
            continue
        shutil.copy2(os.path.join(src_call, name), os.path.join(dst, name))


def gdn(rows):
    pages = rows // 256
    src = os.path.join(R, "dump_gdn16")
    calls = [os.path.join(src, f"call_{30 + i:05d}") for i in range(pages)]
    dst = os.path.join(R, "test", f"big_gdn{rows}")
    fresh(dst)
    copy_inputs(calls[0], dst, skip=("x.bin", "valid_tokens.bin"))
    cat([os.path.join(c, "x.bin") for c in calls], os.path.join(dst, "x.bin"))
    cat([os.path.join(c, "out_residual_out.bin") for c in calls],
        os.path.join(dst, "ref_residual_out.bin"))
    for name in ("state_out", "q_conv_state_out", "k_conv_state_out", "v_conv_state_out"):
        shutil.copy2(os.path.join(calls[-1], f"out_{name}.bin"), os.path.join(dst, f"ref_{name}.bin"))


def attn(rows):
    pages = rows // 256
    src = os.path.join(R, "dump_attn16")
    calls = [os.path.join(src, f"call_{10 + i:05d}") for i in range(pages)]
    dst = os.path.join(R, "test", f"big_attn{rows}")
    fresh(dst)
    copy_inputs(calls[0], dst, skip=("x.bin", "position_ids.bin"))
    cat([os.path.join(c, "x.bin") for c in calls], os.path.join(dst, "x.bin"))
    cat([os.path.join(c, "out_output.bin") for c in calls],
        os.path.join(dst, "ref_output.bin"))
    pos = array.array("i")
    for axis in range(3):
        for c in calls:
            a = array.array("i")
            with open(os.path.join(c, "position_ids.bin"), "rb") as f:
                a.fromfile(f, 3 * 256)
            pos.extend(a[axis * 256:(axis + 1) * 256])
    with open(os.path.join(dst, "position_ids.bin"), "wb") as f:
        pos.tofile(f)


for rows in (3840, 4096):
    gdn(rows)
    attn(rows)
    print("assembled", rows)
