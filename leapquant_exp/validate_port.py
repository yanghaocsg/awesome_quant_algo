# SPDX-License-Identifier: Apache-2.0
"""P2: correctness validation of the pure-PyTorch ports against the reference, using the repository's own judge.

Reuses the exact inputs and judge criterion of <leapquant-kernels>/{gdn,kda}/benchmark.py
(device-parameterized so it runs on CPU or CUDA):
  err = mean |decode(checkpoint) - S| over the head's V x K elements, candidate and reference.
  PASS <=> finite, codes in [-127,127], and for every head:
      err_cand <= (1+REL_TOL)*err_ref   if err_ref >= FLOOR*mean|S|   (tight head)
      err_cand <= CAP*mean|S|           otherwise                      (head the format captures almost exactly)
Runs, for every batch size in workloads.jsonl, three inputs: structured edge-case heads,
the reference's own next-window checkpoint, and unstructured in-domain random.

Usage:
  python validate_port.py --which gdn   [--device cpu|cuda]
  python validate_port.py --which kda
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
KERN = Path(r"C:\code_2026\turboquant量化\leapquant-kernels")
HV, K, V, L, R, QMAX = 32, 128, 128, 16, 4, 127.0
REL_TOL, FLOOR, CAP = 0.01, 1e-4, 1e-3


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode(codes, s_k, s_v, u, q):
    return (torch.einsum("bhrv,bhrk->bhvk", q.double(), u.double())
            + codes.double() * (s_k.double()[..., None, :] / QMAX) * s_v.double()[..., :, None])


# ---------------------------------------------------------------------------------- GDN
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


# ---------------------------------------------------------------------------------- KDA
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


def judge(out, ref, S):
    finite = all(bool(torch.isfinite(t.float()).all()) for t in out[1:]); inrange = bool((out[0].int().abs() <= 127).all())
    e_out = (decode(*out) - S).abs().mean((0, 2, 3)); e_ref = (decode(*ref) - S).abs().mean((0, 2, 3)); mag = S.abs().mean((0, 2, 3))
    tight = (e_ref >= FLOOR * mag) & (e_ref > 0)
    ok = finite and inrange and bool((e_out[tight] <= (1 + REL_TOL) * e_ref[tight]).all()) and bool((e_out[~tight] <= CAP * mag[~tight]).all())
    ratio = (e_out[tight] / e_ref[tight]).max().item() if tight.any() else 0.0
    return ok, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["gdn", "kda"], required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    which = a.which
    df = json.loads((KERN / which / "definition.json").read_text())
    ns = {}; exec(df["reference"], ns); ref_run = ns["run"]
    impl = load_module(HERE / f"{which}_torch.py", f"{which}_torch")
    structured, random, update, exact = (globals()[f"{which}_structured"], globals()[f"{which}_random"],
                                         globals()[f"{which}_update_buffer"], globals()[f"{which}_exact_state"])
    q0_idx = {"gdn": 9, "kda": 10}[which]
    sizes = [json.loads(l)["workload"]["axes"]["batch_size"] for l in (KERN / which / "workloads.jsonl").read_text().splitlines() if l.strip()]

    print(f"{which.upper()} pure-torch port vs reference  |  device={dev}  |  torch {torch.__version__}\n")
    print("| batch | input | PASS | worst-head err ratio (tight heads) |")
    print("| --- | --- | --- | --- |")
    allok = True
    for bs in sizes:
        first = structured(bs, dev, seed=bs)
        ref1 = ref_run(*[t.clone() for t in first])
        second = (*ref1, *update(bs, dev, torch.Generator(device=dev).manual_seed(bs + 1)), first[q0_idx])
        for name, inp, ref in (("structured", first, ref1),
                               ("second_window", second, None),
                               ("random", random(bs, dev, seed=bs + 2), None)):
            ref = ref_run(*[t.clone() for t in inp]) if ref is None else ref
            out = impl.run(*[t.clone() for t in inp])
            ok, ratio = judge(out, ref, exact(inp))
            allok &= ok
            print(f"| {bs} | {name} | {'PASS' if ok else 'FAIL'} | {ratio:.6f} |", flush=True)
    print("\nRESULT:", "ALL PASS" if allok else "SOME FAIL")


if __name__ == "__main__":
    main()
