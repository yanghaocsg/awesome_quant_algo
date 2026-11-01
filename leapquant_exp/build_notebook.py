# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained Colab notebook for the LeapQuant pure-PyTorch port (P1 + P2).

The notebook embeds:
  * the mathematical reference (verbatim from gdn|kda/definition.json) -- the truth to compare against
  * the pure-PyTorch port (verbatim from gdn_torch.py / kda_torch.py)
  * the judge + input construction (mirrors {gdn,kda}/benchmark.py, device-parameterized)
Runs with zero file / network dependencies beyond torch (present in Colab).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERN = Path(r"C:\code_2026\turboquant量化\leapquant-kernels")


def json_of(p):
    return json.loads(p.read_text(encoding="utf-8"))["reference"]


GDN_REF = json_of(KERN / "gdn" / "definition.json")
KDA_REF = json_of(KERN / "kda" / "definition.json")
GDN_PORT = (HERE / "gdn_torch.py").read_text(encoding="utf-8")
KDA_PORT = (HERE / "kda_torch.py").read_text(encoding="utf-8")

RUN_SRC = r'''
# --------------------------------------------------------------------------- judge / inputs (mirrors {gdn,kda}/benchmark.py, device-parameterized)
QMAX, EPS = 127.0, 1e-12
REL_TOL, FLOOR, CAP = 0.01, 1e-4, 1e-3
HV, K, V, L, R = 32, 128, 128, 16, 4

def decode(codes, s_k, s_v, u, q):
    return (torch.einsum("bhrv,bhrk->bhvk", q.double(), u.double())
            + codes.double() * (s_k.double()[..., None, :] / QMAX) * s_v.double()[..., :, None])

def judge(out, ref, S):
    finite = all(bool(torch.isfinite(t.float()).all()) for t in out[1:]); inrange = bool((out[0].int().abs() <= 127).all())
    e_out = (decode(*out) - S).abs().mean((0, 2, 3)); e_ref = (decode(*ref) - S).abs().mean((0, 2, 3)); mag = S.abs().mean((0, 2, 3))
    tight = (e_ref >= FLOOR * mag) & (e_ref > 0)
    ok = finite and inrange and bool((e_out[tight] <= (1 + REL_TOL) * e_ref[tight]).all()) and bool((e_out[~tight] <= CAP * mag[~tight]).all())
    ratio = (e_out[tight] / e_ref[tight]).max().item() if tight.any() else 0.0
    return ok, ratio

def gdn_update_buffer(bs, dev, g):
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    kbuf = (rn(bs, HV, L, K) * 0.3).to(torch.bfloat16); ubuf = (rn(bs, HV, L, V) * 0.3).to(torch.bfloat16)
    w = torch.rand(bs, HV, L, device=dev, generator=g); p = torch.rand(bs, HV, device=dev, generator=g) * 0.5 + 0.5
    return kbuf, ubuf, w, p

def gdn_structured(bs, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    codes = (rn(bs, HV, V, K) * 40).round().clamp(-127, 127).to(torch.int8)
    s_k = torch.rand(bs, HV, K, device=dev, generator=g) * 0.2 + 0.05; s_v = torch.rand(bs, HV, V, device=dev, generator=g) * 0.5 + 0.1
    s_v[..., 5] *= 30
    u = (rn(bs, HV, R, K) * 0.5).to(torch.float16); q = (rn(bs, HV, R, V) * 0.5).to(torch.float16)
    kbuf, ubuf, w, p = gdn_update_buffer(bs, dev, g)
    for h in (1, 2, 3, 6): codes[:, h] = 0; u[:, h] = 0; q[:, h] = 0
    w[:, 1, 1:] = 0; w[:, 2] = 0; w[:, 3, 2:] = 0; w[:, 6, 1:] = 0
    codes[:, 6] = (rn(bs, V, K) * 40).round().clamp(-127, 127).to(torch.int8); s_k[:, 6] *= 1e-6
    s_k[:, 4] *= 1e-4; u[:, 4] *= 1e-2; ubuf[:, 4] *= 1e-4
    s_k[:, 5] *= 3e2; u[:, 5] *= 17; ubuf[:, 5] *= 3e2
    q0 = torch.randn(V, R, generator=torch.Generator().manual_seed(0)).to(dev)
    return codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0

def gdn_random(bs, dev, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    lu = lambda *s: torch.exp(rn(*s) * 0.7)
    return (torch.randint(-127, 128, (bs, HV, V, K), device=dev, generator=g, dtype=torch.int8), lu(bs, HV, K) * 0.05, lu(bs, HV, V) * 0.3,
            rn(bs, HV, R, K, dt=torch.float16), rn(bs, HV, R, V, dt=torch.float16), rn(bs, HV, L, K, dt=torch.bfloat16),
            rn(bs, HV, L, V, dt=torch.bfloat16), torch.rand(bs, HV, L, device=dev, generator=g), torch.rand(bs, HV, device=dev, generator=g) * 0.999 + 0.001, rn(V, R))

def gdn_exact_state(inp):
    codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0 = inp
    return p.double()[..., None, None] * decode(codes, s_k, s_v, u, q) + torch.einsum("bhj,bhjv,bhjk->bhvk", w.double(), ubuf.double(), kbuf.double())

def kda_update_buffer(bs, dev, g):
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    kbuf = (rn(bs, HV, L, K) * 0.3).to(torch.bfloat16); ubuf = (rn(bs, HV, L, V) * 0.3).to(torch.bfloat16)
    w = torch.ones(bs, HV, L, device=dev)
    gbuf = (torch.rand(bs, HV, L, K, device=dev, generator=g) * 0.4 + 0.6).half()
    pcum = gbuf.float().log().sum(-2) - torch.rand(bs, HV, K, device=dev, generator=g) * 0.5
    return kbuf, ubuf, gbuf, w, pcum

def kda_structured(bs, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    codes = (rn(bs, HV, V, K) * 40).round().clamp(-127, 127).to(torch.int8)
    s_k = torch.rand(bs, HV, K, device=dev, generator=g) * 0.2 + 0.05; s_v = torch.rand(bs, HV, V, device=dev, generator=g) * 0.5 + 0.1
    s_v[..., 5] *= 30
    u = (rn(bs, HV, R, K) * 0.5).to(torch.float16); q = (rn(bs, HV, R, V) * 0.5).to(torch.float16)
    kbuf, ubuf, gbuf, w, pcum = kda_update_buffer(bs, dev, g)
    for h in (1, 2, 3, 6): codes[:, h] = 0; u[:, h] = 0; q[:, h] = 0
    w[:, 1, 1:] = 0; w[:, 2] = 0; w[:, 3, 2:] = 0; w[:, 6, 1:] = 0
    codes[:, 6] = (rn(bs, V, K) * 40).round().clamp(-127, 127).to(torch.int8); s_k[:, 6] *= 1e-6
    s_k[:, 4] *= 1e-4; u[:, 4] *= 1e-2; ubuf[:, 4] *= 1e-4
    s_k[:, 5] *= 3e2; u[:, 5] *= 17; ubuf[:, 5] *= 3e2
    pcum[:, 7] -= 200.0
    gbuf[:, 8, :, :32] = 1.0; pcum[:, 8, :32] = 0.0
    q0 = torch.randn(V, R, generator=torch.Generator().manual_seed(0)).to(dev)
    return codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, q0

def kda_random(bs, dev, seed=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    lu = lambda *s: torch.exp(rn(*s) * 0.7)
    return (torch.randint(-127, 128, (bs, HV, V, K), device=dev, generator=g, dtype=torch.int8), lu(bs, HV, K) * 0.05, lu(bs, HV, V) * 0.3,
            rn(bs, HV, R, K, dt=torch.float16), rn(bs, HV, R, V, dt=torch.float16), rn(bs, HV, L, K, dt=torch.bfloat16),
            rn(bs, HV, L, V, dt=torch.bfloat16), (torch.rand(bs, HV, L, K, device=dev, generator=g) * 0.999 + 0.001).half(),
            torch.rand(bs, HV, L, device=dev, generator=g), -torch.rand(bs, HV, K, device=dev, generator=g) * 20, rn(V, R))

def kda_exact_state(inp):
    codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, q0 = inp
    after = torch.flip(torch.cumprod(torch.flip(gbuf.double(), [-2]), -2), [-2])
    D = torch.cat([after[..., 1:, :], torch.ones_like(after[..., :1, :])], -2)
    return decode(codes, s_k, s_v, u, q) * torch.exp(pcum.double())[..., None, :] + torch.einsum("bhj,bhjv,bhjk->bhvk", w.double(), ubuf.double(), kbuf.double() * D)

def run_check(which, port_run, ref_run, sizes, dev):
    structured, random, update, exact = (globals()[f"{which}_structured"], globals()[f"{which}_random"],
                                         globals()[f"{which}_update_buffer"], globals()[f"{which}_exact_state"])
    q0_idx = {"gdn": 9, "kda": 10}[which]
    rows = []
    for bs in sizes:
        first = structured(bs, dev, seed=bs)
        ref1 = ref_run(*[t.clone() for t in first])
        second = (*ref1, *update(bs, dev, torch.Generator(device=dev).manual_seed(bs + 1)), first[q0_idx])
        for name, inp, ref in (("structured", first, ref1), ("second_window", second, None), ("random", random(bs, dev, seed=bs + 2), None)):
            ref = ref_run(*[t.clone() for t in inp]) if ref is None else ref
            out = port_run(*[t.clone() for t in inp])
            ok, ratio = judge(out, ref, exact(inp))
            rows.append((bs, name, ok, ratio))
    return rows
'''

cells = [
    {"cell_type": "markdown", "source": [
        "# LeapQuant Flush — Pure-PyTorch Port (Colab Verification)\n",
        "\n",
        "Self-contained validation of **P1 (pure-torch reference port) + P2 (repository judge)** for the LeapQuant window-boundary flush kernel.\n",
        "\n",
        "- **Reference**: verbatim from `gdn|kda/definition.json` (fp32 math truth).\n",
        "- **Port**: `gdn_torch.py` / `kda_torch.py` (literal translation, no cutlass/tilelang/B300).\n",
        "- **Judge**: mirrors `{gdn,kda}/benchmark.py` — decode the new checkpoint back to a V×K state and compare, head by head, against the fp64-rebuilt state `S`. PASS ⇔ finite, codes in [-127,127], and `err_cand ≤ (1+1%)·err_ref` (or `≤1e-3·|S|` when the reference error is below `1e-4·|S|`).\n",
        "\n",
        "> Only dependency is `torch`, which Colab ships with (GPU runtime recommended). Zero file/network access needed."
    ]},
    {"cell_type": "code", "source": [
        "import torch\n",
        "DEV = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(\"torch\", torch.__version__, \"| device\", DEV, \"|\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")\n",
        "BATCHES = [16, 32, 64, 128]   # add 256, 512 to match the repository workloads (slower on CPU)"
    ]},
    {"cell_type": "markdown", "source": [
        "## 1. Load the mathematical reference (verbatim from definition.json)"
    ]},
    {"cell_type": "code", "source": [
        "GDN_REF = '''" + GDN_REF + "'''\n",
        "KDA_REF = '''" + KDA_REF + "'''\n",
        "_g = {}; exec(GDN_REF, _g); ref_gdn = _g[\"run\"]\n",
        "_k = {}; exec(KDA_REF, _k); ref_kda = _k[\"run\"]\n",
        "print(\"reference loaded: GDN, KDA\")"
    ]},
    {"cell_type": "markdown", "source": [
        "## 2. Pure-PyTorch port (P1)"
    ]},
    {"cell_type": "code", "source": [
        "_gport = {}; exec('''" + GDN_PORT + "''', _gport)\n",
        "gdn_torch_run = _gport[\"run\"]\n",
        "print(\"gdn port loaded\")"
    ]},
    {"cell_type": "code", "source": [
        "_kport = {}; exec('''" + KDA_PORT + "''', _kport)\n",
        "kda_torch_run = _kport[\"run\"]\n",
        "print(\"kda port loaded\")"
    ]},
    {"cell_type": "markdown", "source": [
        "## 3. Judge + input construction (P2, mirrors benchmark.py)"
    ]},
    {"cell_type": "code", "source": [RUN_SRC]},
    {"cell_type": "markdown", "source": [
        "## 4. Run validation (GDN, then KDA)"
    ]},
    {"cell_type": "code", "source": [
        "print(\"=\" * 62, \"\\nGDN pure-torch port vs reference\\n\", \"=\" * 62)\n",
        "for bs, name, ok, ratio in run_check(\"gdn\", gdn_torch_run, ref_gdn, BATCHES, DEV):\n",
        "    print(f\"| {bs:>4} | {name:<13} | {'PASS' if ok else 'FAIL':<5} | worst-head err ratio {ratio:.6f}\")\n",
        "print(\"\\n\" + \"=\" * 62, \"\\nKDA pure-torch port vs reference\\n\", \"=\" * 62)\n",
        "for bs, name, ok, ratio in run_check(\"kda\", kda_torch_run, ref_kda, BATCHES, DEV):\n",
        "    print(f\"| {bs:>4} | {name:<13} | {'PASS' if ok else 'FAIL':<5} | worst-head err ratio {ratio:.6f}\")"
    ]},
    {"cell_type": "markdown", "source": [
        "## 5. Interpreting the result\n",
        "\n",
        "- `worst-head err ratio = 1.000000` means the port reproduces the fp32 reference *exactly* (it is a faithful direct translation), so it passes the judge with maximum slack.\n",
        "- `PASS` across every batch × input family = the pure-torch port is contract- and numerically-equivalent to the repository's reference on all cases (`3·len(BATCHES)` here, or all 18 when BATCHES = the 6 workloads).\n",
        "- This validates P1/P2 on any torch device; for production performance (P3+) the port would need `torch.compile` / custom kernels, per the feasibility note."
    ]},
]

# Colab requires every cell to carry metadata.id (and code cells outputs/execution_count);
# missing them makes the frontend crash with "Cannot read properties of undefined (reading 'id')".
for _i, _c in enumerate(cells):
    _c.setdefault("metadata", {})
    _c["metadata"].setdefault("id", f"kern_{_i}")
    if _c["cell_type"] == "code":
        _c.setdefault("outputs", [])
        _c.setdefault("execution_count", None)

nb = {
    "nbformat": 4, "nbformat_minor": 2,
    "metadata": {"colab": {"name": "leapquant_torch_port.ipynb", "provenance": [], "toc_visible": True},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3"},
                 "accelerator": "GPU"},
    "cells": cells,
}
out = HERE / "leapquant_torch_port.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, out.stat().st_size, "bytes")
