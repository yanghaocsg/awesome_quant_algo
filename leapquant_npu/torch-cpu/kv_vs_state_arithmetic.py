#!/usr/bin/env python3
"""kv_vs_state_arithmetic.py -- KV cache vs recurrent-state memory, computed not asserted.

Companion to `deltanet_reading_note.md` (sec 2.2) and `gdn.md` / `kda.md`. The claim
"DeltaNet saves memory" has no fixed answer: it depends on head geometry, on whether the
full-attention layers use GQA, on how many layers are linear, and on the state's dtype.
This script computes the crossover and the per-length numbers so the table in the note can
be reproduced (or overturned) for any model shape.

Model of the world (deliberately simple, stated so it can be argued with):

    KV cache, per layer per token : 2 * kv_heads * head_dim * kv_bytes      (K and V)
    GDN/KDA state, per layer      : q_heads * head_dim * head_dim * state_bytes
    hybrid stack at length T      : L_lin * state + L_full * kv(T)
    full-attention stack at T     : L * kv(T)

Consequence (closed form): the two stacks are equal at

    T* = state_bytes_per_layer / kv_bytes_per_layer_per_token

which, for a plain MHA baseline with everything in bf16, simplifies to head_dim / 2
(64 tokens at d=128) -- and scales by q_heads/kv_heads when the full-attention layers use
GQA. Below T* the fixed state is the *larger* of the two.

Not modelled (on purpose): conv-mixing state, allocator / page padding, CUDA-graph and
scheduler overheads, prefill scratch (chunkwise WY buffers), and the fact that a 3:1 hybrid
also pays the full-attention layers' KV. Add `--conv-kernel` to at least price the conv
state; everything else is left to the reader rather than silently baked in.

Usage:
    python kv_vs_state_arithmetic.py
    python kv_vs_state_arithmetic.py --q-heads 16 --kv-heads 2 --layers 48
    python kv_vs_state_arithmetic.py --kv-heads 16 --lengths 64 512 2048 8192 32768
    python kv_vs_state_arithmetic.py --state-format leapquant --rank 4
    python kv_vs_state_arithmetic.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
STATE_FORMATS = ("fp32", "fp16", "bf16", "int8", "leapquant")


def human(n: float) -> str:
    for unit, scale in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}" if unit == "GiB" else f"{n / scale:.1f} {unit}"
    return f"{n:.0f} B"


def state_bytes_per_layer(
    q_heads: int, head_dim: int, state_format: str, rank: int, conv_kernel: int, conv_bytes: int
) -> tuple:
    """Bytes of one linear layer's recurrent state (ALL of its heads), plus a breakdown."""
    d = head_dim
    if state_format == "leapquant":
        # LeapQuant checkpoint per head (see state.py::encode): int8 payload + per-value
        # scale + per-key smoothing + rank-r compensator tokens held in fp16.
        payload = d * d
        scales = 4 * d + 4 * d
        comp = 2 * (d * rank * 2)
        parts = {"payload(int8)": payload, "scales(fp32)": scales, f"comp(r={rank},fp16)": comp}
        per_head = payload + scales + comp
    elif state_format in DTYPE_BYTES:
        per_head = d * d * DTYPE_BYTES[state_format]
        parts = {f"state({state_format})": per_head}
    else:
        raise ValueError(f"unknown state format {state_format!r}")

    if conv_kernel > 0:
        # causal conv over q, k, v keeps (kernel-1) past steps per channel, per head
        parts[f"conv({conv_kernel}-1,{conv_bytes}B)"] = (conv_kernel - 1) * 3 * d * conv_bytes
        per_head += (conv_kernel - 1) * 3 * d * conv_bytes

    total = per_head * q_heads
    parts = {k: v * q_heads for k, v in parts.items()}
    return total, parts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=48, help="total transformer blocks (default 48, Qwen3-Next 80B shape)")
    p.add_argument("--q-heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=2, help="KV heads in the full-attention layers (== q-heads for plain MHA)")
    p.add_argument("--head-dim", type=int, default=128, help="head dim of the LINEAR (state) layers")
    p.add_argument("--kv-head-dim", type=int, default=None, help="head dim of the full-attention layers (default: --head-dim)")
    p.add_argument("--linear-ratio", type=int, default=3, help="linear layers per one full-attention layer (3 means 3:1)")
    p.add_argument("--all-linear", action="store_true", help="stack is 100%% linear layers (head-to-head vs an all-attention stack)")
    p.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    p.add_argument("--state-format", choices=STATE_FORMATS, default="fp32")
    p.add_argument("--rank", type=int, default=4, help="compensator rank r (state-format=leapquant only)")
    p.add_argument("--conv-kernel", type=int, default=0, help="short-conv kernel size; 0 = ignore (default)")
    p.add_argument("--conv-dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    p.add_argument("--lengths", type=int, nargs="+", default=None)
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    args = p.parse_args(argv)

    if args.layers <= 0 or args.q_heads <= 0 or args.kv_heads <= 0 or args.head_dim <= 0:
        p.error("layers, heads and head-dim must be positive")
    if args.linear_ratio < 0:
        p.error("--linear-ratio must be >= 0")
    kv_dim = args.kv_head_dim or args.head_dim
    if kv_dim <= 0:
        p.error("--kv-head-dim must be positive")

    l_full = args.layers // (args.linear_ratio + 1) if args.linear_ratio >= 0 else args.layers
    if args.all_linear:
        l_full = 0
        l_lin = args.layers
    else:
        if l_full < 1:
            l_full = 1
        l_lin = args.layers - l_full

    kv_b = DTYPE_BYTES[args.kv_dtype]
    conv_b = DTYPE_BYTES[args.conv_dtype]
    kv_per_tok = 2 * args.kv_heads * kv_dim * kv_b
    state_per_layer, parts = state_bytes_per_layer(
        args.q_heads, args.head_dim, args.state_format, args.rank, args.conv_kernel, conv_b
    )

    lengths = args.lengths or [64, 512, 2048, 4096, 8192, 20000, 32768]

    def stack_bytes(t: int) -> tuple:
        kv = t * kv_per_tok
        full_stack = args.layers * kv
        hybrid = l_lin * state_per_layer + l_full * kv
        return full_stack, hybrid

    crossover_exact = state_per_layer / kv_per_tok
    # independent numeric check: smallest integer T where hybrid stops being larger
    lo, hi = 1, max(2, int(math.ceil(crossover_exact * 4)) + 64)
    if stack_bytes(hi)[1] > stack_bytes(hi)[0]:
        print("warning: no crossover found within the search window", file=sys.stderr)
        saved_ge = -1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if stack_bytes(mid)[1] <= stack_bytes(mid)[0]:
                hi = mid
            else:
                lo = mid + 1
        saved_ge = lo

    rows = []
    for t in lengths:
        full_stack, hybrid = stack_bytes(t)
        rows.append(
            {
                "tokens": t,
                "full_stack_bytes": full_stack,
                "hybrid_bytes": hybrid,
                "ratio": (full_stack / hybrid) if hybrid else None,
                "saved_pct": 100.0 * (1 - hybrid / full_stack) if full_stack else None,
            }
        )

    payload = {
        "config": {
            "layers": args.layers,
            "linear_layers": l_lin,
            "full_layers": l_full,
            "linear_ratio": f"{args.linear_ratio}:1",
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "kv_head_dim": kv_dim,
            "kv_dtype": args.kv_dtype,
            "state_format": args.state_format,
            "conv_kernel": args.conv_kernel,
        },
        "per_layer": {
            "kv_bytes_per_token": kv_per_tok,
            "state_bytes": state_per_layer,
            "state_breakdown": parts,
        },
        "crossover_tokens": crossover_exact,
        "crossover_bisection": saved_ge,
        "crossover_agrees": saved_ge == int(math.ceil(crossover_exact)),
        "rows": rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if payload["crossover_agrees"] else 1

    print("kv_vs_state_arithmetic.py")
    print(
        f"  shape      : {args.layers} layers = {l_lin} linear + {l_full} full-attention "
        f"({args.linear_ratio}:1), {args.q_heads} q-heads / {args.kv_heads} kv-heads, "
        f"d_state={args.head_dim}, d_kv={kv_dim}"
    )
    print(f"  KV dtype   : {args.kv_dtype} ({kv_b} B)  -> {kv_per_tok} B per layer per token")
    print(f"  state      : {args.state_format}, {state_per_layer} B per linear layer"
          + (f" (conv kernel {args.conv_kernel} included)" if args.conv_kernel else ""))
    print("               " + ", ".join(f"{k}={v} B" for k, v in parts.items()))
    print()
    print("  T tokens | full-attn stack |   hybrid stack | full/hybrid | saved")
    print("  ---------+-----------------+----------------+-------------+-------")
    for r in rows:
        ratio = f"{r['ratio']:.2f}x" if r["ratio"] else "n/a"
        saved = f"{r['saved_pct']:.1f}%" if r["saved_pct"] is not None else "n/a"
        print(
            f"  {r['tokens']:8d} | {human(r['full_stack_bytes']):>15} | "
            f"{human(r['hybrid_bytes']):>14} | {ratio:>11} | {saved:>6}"
        )
    print()
    print(f"  crossover T* = state_bytes / kv_bytes_per_token = {crossover_exact:.2f} tokens")
    print(f"  independent bisection      = {saved_ge} tokens  -> "
          f"{'MATCH' if payload['crossover_agrees'] else 'MISMATCH'}")
    print(f"  (below T* the recurrent state is the *larger* of the two; above it the hybrid wins)")
    if args.state_format != "fp32":
        ref = args.q_heads * args.head_dim * args.head_dim * 4
        print(f"  state vs fp32 baseline     : {ref} B -> {state_per_layer} B ({ref / state_per_layer:.2f}x smaller)")
    return 0 if payload["crossover_agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
