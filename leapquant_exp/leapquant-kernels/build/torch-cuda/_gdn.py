# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL LeapQuant GDN window flush for SM100/SM103 (tcgen05 design).

One persistent CTA per SM: a TMA producer warp and four consumer groups of four warps. The producer heads a
warpgroup of its own (three idle warps) that gives registers to the consumers with setmaxnreg (32 / 112).
Each group flushes one (slot, head) at a time:
  * TMA brings the checkpoint (int8 codes 128B-swizzled, scales, factors), the update buffer
    and its weights into a 32 KB stage; the stage is released as soon as the last product has read it.
    The small inputs are issued first on the stage barrier and the 16 KB code tile on its own barrier,
    so a group starts a program on the small inputs and waits for the codes only before dequantizing.
  * The state S (fp32, scaled by a per-head power of two 2^sig) is built in tensor memory: one tf32
    MMA writes the scale product 2^8 (p s_v[v] / 127) s_k[k], the dequantization multiplies it by the
    codes, and the rank-16 update (bf16 hi/lo) and the old rank-4 part (tf32 hi/lo) are accumulated by
    tcgen05.mma. Codes become floats as fp16 pairs 2^s (1024 + c + 128) (one PRMT per two codes) converted
    by add.rn.f32.f16 (FHADD, FMA pipe) with -1152 2^s, s = sig - 8: 2^s c exactly, so the ALU pipe only
    sees the byte permutes; a program with s outside the fp16 range [-24, 4] first rescales the scale
    product in TMEM by an exact power of two. Other format conversions also run on the FMA pipe.
  * An fp16 copy of S in shared memory feeds the three subspace-iteration products on the
    tensor cores (as K-major A for S@q and MN-major A for S^T@q); an eight-column projection uses TMEM
    columns 112-119 while their original fp32 state values stay in registers. Q operands carry fp16
    high/low parts. CholeskyQR2 uses FP32x2 mul/FMA for independent factor entries on larger launches
    and the original scalar factorization on smaller launches. It runs on all four warps (one row per thread); its second pass drops
    numerically dependent columns so the result stays orthonormal.
  * E = S - q^T u is formed in place in tensor memory (tf32 MMA with the stored fp16 factors);
    column scales, row scales and int8 codes are computed from fp32 E; code tiles are staged in the
    group's operand buffers and written by per-warp TMA stores.
  * The group's leader warp issues every MMA sequence (also those of the rare precision path) from inline PTX
    (mma8_f16 / mma_list): shared-memory descriptors are (address >> 4) | constant, plus one 64-bit add per K
    step, followed by the commit. Each operand handoff first synchronizes all four writer warps, then the
    leader fences its acquired generic shared-memory view into the async proxy before issuing the MMA.
GPU-side due selection: the producer scans (idx, fill counter) and skips slot 0.
At 16 or more heads per SM, the initial scan lets CTAs with no due heads skip tensor-memory allocation,
consumer setup and empty-stage handoffs. Consumer warp zero checks the first candidate alongside the producer;
a due seed allocates tensor memory before the initial barrier. Otherwise allocation overlaps q0 setup after
the scan, and empty CTAs allocate nothing. The deallocating warp reloads the immutable shared due mask at exit
so it need not stay in a register across the persistent loop. Smaller calls use the original setup. Both paths
are compiled by Pool construction; flush selects using only the public launch size and SM count.
Idle warps of the producer warpgroup prefetch the four TMA descriptors, q0 and (one slice per CTA) the fill-counter array
into L2 at entry, so the first copies of each SM do not wait for a cold descriptor fetch and the due check's
dependent counter loads hit L2.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float16, BFloat16, Float32, Int8, Int32, Uint32, Uint64, Int64, Boolean
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass.utils import SmemAllocator
from cutlass.cute.nvgpu import tcgen05
import cuda.bindings.driver as cuda

HV, K, V, L, R = 32, 128, 128, 16, 4
SS, OFF = 33 * V * K, 12288                    # pool geometry in fp32 words
NG = 4                                         # consumer groups
NSTAGE = 4
assert NSTAGE == NG                            # each group owns one stage
PROD_WARP = 4 * NG
NTHREADS = 32 * (4 * NG + 4)                   # the producer warp heads a warpgroup of its own (setmaxnreg)
REG_CONS, REG_PROD = 112, 32                    # per SMSP: 4 * 32 * 112 + 32 * 32 < 16384 registers (allocation needs slack)
STAGE = 32768
ST_CODES, ST_SC, ST_KB, ST_UB, ST_W, ST_P = 0, 16384, 19456, 23552, 27648, 27712
LOAD_BYTES = 3072 + 4096 + 4096 + 64 + 16           # small inputs (the 16 KB code tile has its own barrier)
EPS = 1e-12
TAU2 = 1e-3                                    # second CholeskyQR pass: drop columns with pivot <= TAU2 * |column|^2
TAU1 = 0.03                                    # first pass: all relative pivots above TAU1 -> kappa^2 <~ 4 / TAU1 ~ 133:
                                               # the first pass is orthonormal to ~1e-5 << fp16 rounding: no second pass
MAGIC_RND = 12582912.0                         # 1.5 * 2^23: add, then the low byte is the rounded integer


# ----------------------------------------------------------------------------------------------
# inline PTX helpers
# ----------------------------------------------------------------------------------------------
def _asm(ret_types, args, text, cons, side=True, loc=None, ip=None):
    if not ret_types:
        llvm.inline_asm(None, args, text, cons, has_side_effects=side, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        return []
    rt = ret_types[0] if len(ret_types) == 1 else llvm.StructType.get_literal(ret_types)
    out = llvm.inline_asm(rt, args, text, cons, has_side_effects=side, is_align_stack=False,
                          asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    if len(ret_types) == 1:
        return [out]
    return [llvm.extractvalue(ret_types[i], out, [i], loc=loc, ip=ip) for i in range(len(ret_types))]


def _u(x, loc, ip):
    return Uint32(x).ir_value(loc=loc, ip=ip)


def _f(x, loc, ip):
    return Float32(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def tmem_alloc(dst_smem_addr, ncols: cutlass.Constexpr, *, loc=None, ip=None):
    _asm([], [_u(dst_smem_addr, loc, ip)],
         f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {ncols};", "r", loc=loc, ip=ip)


@dsl_user_op
def tmem_free(addr, ncols: cutlass.Constexpr, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)],
         f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {ncols};", "r", loc=loc, ip=ip)


@dsl_user_op
def tmem_relinquish(*, loc=None, ip=None):
    _asm([], [], "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def tmem_ld8(addr, *, loc=None, ip=None):
    r = _asm([T.f32()] * 8, [_u(addr, loc, ip)],
             "tcgen05.ld.sync.aligned.32x32b.x8.b32 {$0,$1,$2,$3,$4,$5,$6,$7}, [$8]; tcgen05.wait::ld.sync.aligned;",
             ",".join(["=f"] * 8 + ["r"]), loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def tmem_ld4(addr, *, loc=None, ip=None):
    r = _asm([T.f32()] * 4, [_u(addr, loc, ip)],
             "tcgen05.ld.sync.aligned.32x32b.x4.b32 {$0,$1,$2,$3}, [$4]; tcgen05.wait::ld.sync.aligned;",
             ",".join(["=f"] * 4 + ["r"]), loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def tmem_ld16_nw(addr, *, loc=None, ip=None):
    """Issue a 32x32b.x16 TMEM load (no wait); pass the result through tmem_wait_ld before use."""
    r = _asm([T.f32()] * 16, [_u(addr, loc, ip)],
             "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + ",".join("$" + str(i) for i in range(16)) + "}, [$16];",
             ",".join(["=f"] * 16 + ["r"]), loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def tmem_ld32_nw(addr, *, loc=None, ip=None):
    """Issue a 32x32b.x32 TMEM load (no wait); pass the result through tmem_wait_ld before use."""
    r = _asm([T.f32()] * 32, [_u(addr, loc, ip)],
             "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + ",".join("$" + str(i) for i in range(32)) + "}, [$32];",
             ",".join(["=f"] * 32 + ["r"]), loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def tmem_ld16x256x4_nw(addr, *, loc=None, ip=None):
    r = _asm([T.f32()] * 16, [_u(addr, loc, ip)],
             "tcgen05.ld.sync.aligned.16x256b.x4.b32 {" + ",".join("$" + str(i) for i in range(16)) + "}, [$16];",
             ",".join(["=f"] * 16 + ["r"]), loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def tmem_wait_ld(vals, *, loc=None, ip=None):
    """tcgen05.wait::ld with a register dependency on the loaded values (uses cannot move above the wait)."""
    n = len(vals)
    r = _asm([T.f32()] * n, [_f(x, loc, ip) for x in vals], "tcgen05.wait::ld.sync.aligned;",
             ",".join(["=f"] * n + [str(i) for i in range(n)]), loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def tmem_st16_nw(addr, vals, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)] + [_f(x, loc, ip) for x in vals],
         "tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {" + ",".join("$" + str(i + 1) for i in range(16)) + "};",
         ",".join(["r"] + ["f"] * 16), loc=loc, ip=ip)


@dsl_user_op
def tmem_st32_nw(addr, vals, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)] + [_f(x, loc, ip) for x in vals],
         "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {" + ",".join("$" + str(i + 1) for i in range(32)) + "};",
         ",".join(["r"] + ["f"] * 32), loc=loc, ip=ip)


@dsl_user_op
def tmem_st8_nw(addr, vals, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)] + [_f(x, loc, ip) for x in vals],
         "tcgen05.st.sync.aligned.32x32b.x8.b32 [$0], {" + ",".join("$" + str(i + 1) for i in range(8)) + "};",
         ",".join(["r"] + ["f"] * 8), loc=loc, ip=ip)


@dsl_user_op
def mma8_f16(d, a4, b4, mbar, aconst: cutlass.Constexpr, aoffs: cutlass.Constexpr, bconst: cutlass.Constexpr,
             boffs: cutlass.Constexpr, idesc: cutlass.Constexpr, acc_first: cutlass.Constexpr = False, *, loc=None, ip=None):
    """Eight tcgen05.mma kind::f16 steps (first one overwrites D) with smem descriptors (addr >> 4) | const plus
    per-step offsets, then a commit to mbar; issue from one elected thread."""
    txt = ["{", ".reg .b64 a0, b0, ad, bd;", ".reg .b32 id;", ".reg .pred pf, pt;",
           "setp.ne.b32 pf, $1, $1;", "setp.eq.b32 pt, $1, $1;", f"mov.b32 id, {idesc};",
           "cvt.u64.u32 a0, $1;", f"or.b64 a0, a0, {aconst};", "cvt.u64.u32 b0, $2;", f"or.b64 b0, b0, {bconst};",
           f"tcgen05.mma.cta_group::1.kind::f16 [$0], a0, b0, id, {'pt' if acc_first else 'pf'};"]
    for kb in range(1, 8):
        txt += [f"add.s64 ad, a0, {aoffs[kb]};", f"add.s64 bd, b0, {boffs[kb]};",
                "tcgen05.mma.cta_group::1.kind::f16 [$0], ad, bd, id, pt;"]
    txt += ["tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$3];", "}"]
    _asm([], [_u(d, loc, ip), _u(a4, loc, ip), _u(b4, loc, ip), _u(mbar, loc, ip)], " ".join(txt), "r,r,r,r",
         loc=loc, ip=ip)


@dsl_user_op
def mma_list(d, mbar, bases, steps, *, loc=None, ip=None):
    """tcgen05.mma steps (kind, a base index, a const, b base index, b const, idesc, accumulate) with smem descriptors
    (bases[i] | const) (bases are addr >> 4), then a commit to mbar; issue from one elected thread."""
    n = len(bases)
    txt = ["{", ".reg .b64 ad, bd;", ".reg .b32 id;", ".reg .pred pf, pt;"] + [f".reg .b64 x{i};" for i in range(n)]
    txt += ["setp.ne.b32 pf, $0, $0;", "setp.eq.b32 pt, $0, $0;"] + [f"cvt.u64.u32 x{i}, ${2 + i};" for i in range(n)]
    for kind, ai, ac, bi, bc, idesc, acc in steps:
        txt += [f"or.b64 ad, x{ai}, {ac};", f"or.b64 bd, x{bi}, {bc};", f"mov.b32 id, {idesc};",
                f"tcgen05.mma.cta_group::1.kind::{kind} [$0], ad, bd, id, {'pt' if acc else 'pf'};"]
    txt += ["tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$1];", "}"]
    _asm([], [_u(d, loc, ip), _u(mbar, loc, ip)] + [_u(b, loc, ip) for b in bases], " ".join(txt),
         ",".join(["r"] * (2 + n)), loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None):
    _asm([], [], "tcgen05.wait::st.sync.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def tc_before(*, loc=None, ip=None):
    _asm([], [], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def tc_after(*, loc=None, ip=None):
    _asm([], [], "tcgen05.fence::after_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_async_smem(*, loc=None, ip=None):
    _asm([], [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def sts128(addr, a, b, c, d, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)] + [_u(x, loc, ip) for x in (a, b, c, d)],
         "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc=loc, ip=ip)


@dsl_user_op
def sts64(addr, a, b, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip), _u(a, loc, ip), _u(b, loc, ip)], "st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", loc=loc, ip=ip)


@dsl_user_op
def sts32(addr, a, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip), _u(a, loc, ip)], "st.shared.b32 [$0], $1;", "r,r", loc=loc, ip=ip)


@dsl_user_op
def lds128(addr, *, loc=None, ip=None):
    r = _asm([T.i32()] * 4, [_u(addr, loc, ip)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc=loc, ip=ip)
    return [Uint32(x) for x in r]


@dsl_user_op
def lds128f(addr, *, loc=None, ip=None):
    r = _asm([T.f32()] * 4, [_u(addr, loc, ip)], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", loc=loc, ip=ip)
    return [Float32(x) for x in r]


@dsl_user_op
def lds32(addr, *, loc=None, ip=None):
    r = _asm([T.i32()], [_u(addr, loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def lds32f(addr, *, loc=None, ip=None):
    r = _asm([T.f32()], [_u(addr, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def lds16(addr, *, loc=None, ip=None):
    r = _asm([T.i32()], [_u(addr, loc, ip)], "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.u32.u16 $0, h; }", "=r,r", loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def stg32(addr, a, *, loc=None, ip=None):
    _asm([], [Int64(addr).ir_value(loc=loc, ip=ip), _u(a, loc, ip)], "st.global.b32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def stg16(addr, a, *, loc=None, ip=None):
    _asm([], [Int64(addr).ir_value(loc=loc, ip=ip), _u(a, loc, ip)],
         "{ .reg .b16 h; cvt.u16.u32 h, $1; st.global.b16 [$0], h; }", "l,r", loc=loc, ip=ip)


@dsl_user_op
def stg16_hi(addr, a, *, loc=None, ip=None):
    _asm([], [Int64(addr).ir_value(loc=loc, ip=ip), _u(a, loc, ip)],
         "{ .reg .b16 l, h; mov.b32 {l, h}, $1; st.global.b16 [$0], h; }", "l,r", loc=loc, ip=ip)


@dsl_user_op
def ldg32(addr, *, loc=None, ip=None):
    r = _asm([T.i32()], [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def ldg32_volatile(addr, *, loc=None, ip=None):
    r = _asm([T.i32()], [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.relaxed.gpu.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def f2h2(a, b, *, loc=None, ip=None):
    """fp16x2 with a in the low half."""
    r = _asm([T.i32()], [_f(a, loc, ip), _f(b, loc, ip)], "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f", side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def f2bf2(a, b, *, loc=None, ip=None):
    r = _asm([T.i32()], [_f(a, loc, ip), _f(b, loc, ip)], "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def h2f2(x, *, loc=None, ip=None):
    r = _asm([T.f32(), T.f32()], [_u(x, loc, ip)],
             "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; add.rn.f32.f16 $0, lo, 0f80000000; add.rn.f32.f16 $1, hi, 0f80000000; }", "=f,=f,r",
             side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def h2f(x, *, loc=None, ip=None):
    r = _asm([T.f32()], [_u(x, loc, ip)], "{ .reg .b16 lo; cvt.u16.u32 lo, $1; add.rn.f32.f16 $0, lo, 0f80000000; }", "=f,r",
             side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def f2h(a, *, loc=None, ip=None):
    r = _asm([T.i32()], [_f(a, loc, ip)], "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.u32.u16 $0, h; }", "=r,f",
             side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def f2tf32(a, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(a, loc, ip)], "{ .reg .b32 t; cvt.rna.tf32.f32 t, $1; mov.b32 $0, t; }", "=f,f",
             side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def fmul2(a0, a1, b0, b1, *, loc=None, ip=None):
    r = _asm([T.f32(), T.f32()], [_f(x, loc, ip) for x in (a0, a1, b0, b1)],
             "{ .reg .b64 x, y, z; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $5}; mul.rn.f32x2 z, x, y; mov.b64 {$0, $1}, z; }",
             "=f,=f,f,f,f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def ffma2(a0, a1, b0, b1, c0, c1, *, loc=None, ip=None):
    r = _asm([T.f32(), T.f32()], [_f(x, loc, ip) for x in (a0, a1, b0, b1, c0, c1)],
             "{ .reg .b64 x, y, z, w; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $5}; mov.b64 z, {$6, $7}; fma.rn.f32x2 w, x, y, z; mov.b64 {$0, $1}, w; }",
             "=f,=f,f,f,f,f,f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def fsub2(a0, a1, b0, b1, *, loc=None, ip=None):
    r = _asm([T.f32(), T.f32()], [_f(x, loc, ip) for x in (a0, a1, b0, b1)],
             "{ .reg .b64 x, y, z; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $5}; sub.rn.f32x2 z, x, y; mov.b64 {$0, $1}, z; }",
             "=f,=f,f,f,f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def fadd2(a0, a1, b0, b1, *, loc=None, ip=None):
    r = _asm([T.f32(), T.f32()], [_f(x, loc, ip) for x in (a0, a1, b0, b1)],
             "{ .reg .b64 x, y, z; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $5}; add.rn.f32x2 z, x, y; mov.b64 {$0, $1}, z; }",
             "=f,=f,f,f,f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def hadd2(a, b, *, loc=None, ip=None):
    r = _asm([T.i32()], [_u(a, loc, ip), _u(b, loc, ip)], "add.rn.f16x2 $0, $1, $2;", "=r,r,r", side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def fhadd2(h, c, *, loc=None, ip=None):
    """(f32(h.lo) + c, f32(h.hi) + c) for an f16x2 register h (two FHADD)."""
    r = _asm([T.f32(), T.f32()], [_u(h, loc, ip), _f(c, loc, ip)],
             "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; add.rn.f32.f16 $0, lo, $3; add.rn.f32.f16 $1, hi, $3; }",
             "=f,=f,r,f", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def bf2f2(x, *, loc=None, ip=None):
    """bf16x2 -> two fp32 (exact), FHADD.BF16 with -0.0."""
    r = _asm([T.f32(), T.f32()], [_u(x, loc, ip)],
             "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; add.rn.f32.bf16 $0, lo, 0f80000000; add.rn.f32.bf16 $1, hi, 0f80000000; }",
             "=f,=f,r", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def bfsub2(h, a, b, *, loc=None, ip=None):
    """(f32(h.lo) - a, f32(h.hi) - b) for a bf16x2 register h (exact when h holds truncations of a, b)."""
    r = _asm([T.f32(), T.f32()], [_u(h, loc, ip), _f(a, loc, ip), _f(b, loc, ip)],
             "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $2; sub.rn.f32.bf16 $0, lo, $3; sub.rn.f32.bf16 $1, hi, $4; }",
             "=f,=f,r,f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def maxabs3(m, a, b, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(m, loc, ip), _f(a, loc, ip), _f(b, loc, ip)],
             "{ .reg .f32 x, y; abs.f32 x, $2; abs.f32 y, $3; max.f32 $0, $1, x, y; }", "=f,f,f,f",
             side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def bmaxabs2(a, b, *, loc=None, ip=None):
    """bf16x2 lanes: max(|a|, |b|) (sign bits = xor of the signs; mask them off)."""
    r = _asm([T.i32()], [_u(a, loc, ip), _u(b, loc, ip)], "max.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r",
             side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def prmt(a, b, sel, *, loc=None, ip=None):
    r = _asm([T.i32()], [_u(a, loc, ip), _u(b, loc, ip), _u(sel, loc, ip)], "prmt.b32 $0, $1, $2, $3;", "=r,r,r,r",
             side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def redux_max(x, *, loc=None, ip=None):
    r = _asm([T.i32()], [_u(x, loc, ip)], "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r", side=False, loc=loc, ip=ip)
    return Uint32(r[0])


@dsl_user_op
def shfl_bfly(x, m: cutlass.Constexpr, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(x, loc, ip)], f"shfl.sync.bfly.b32 $0, $1, {m}, 0x1f, 0xffffffff;", "=f,f",
             side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def shfl_idx(x, src, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(x, loc, ip), Int32(src).ir_value(loc=loc, ip=ip)],
             "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r", side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def rsqrt_approx(x, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(x, loc, ip)], "rsqrt.approx.ftz.f32 $0, $1;", "=f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def rcp_approx(x, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(x, loc, ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def rcp_rn(x, *, loc=None, ip=None):
    r = _asm([T.f32()], [_f(x, loc, ip)], "rcp.rn.f32 $0, $1;", "=f,f", side=False, loc=loc, ip=ip)
    return Float32(r[0])


@dsl_user_op
def nanosleep_ns(t: cutlass.Constexpr, *, loc=None, ip=None):
    _asm([], [], f"nanosleep.u32 {t};", "", loc=loc, ip=ip)


@dsl_user_op
def exp2_int(e, *, loc=None, ip=None):
    """2^e as fp32 for integer e in [-126, 127]."""
    v = (Int32(e) + Int32(127)).to(Uint32) << Uint32(23)
    return v.bitcast(Float32)


@dsl_user_op
def tma_load_4d(dst, tmap, c0, c1, c2, c3, mbar, *, loc=None, ip=None):
    _asm([], [_u(dst, loc, ip), Int64(tmap).ir_value(loc=loc, ip=ip)] + [Int32(c).ir_value(loc=loc, ip=ip) for c in (c0, c1, c2, c3)] + [_u(mbar, loc, ip)],
         "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4, $5}], [$6];",
         "r,l,r,r,r,r,r", loc=loc, ip=ip)


@dsl_user_op
def tma_load_3d(dst, tmap, c0, c1, c2, mbar, *, loc=None, ip=None):
    _asm([], [_u(dst, loc, ip), Int64(tmap).ir_value(loc=loc, ip=ip)] + [Int32(c).ir_value(loc=loc, ip=ip) for c in (c0, c1, c2)] + [_u(mbar, loc, ip)],
         "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, $4}], [$5];",
         "r,l,r,r,r,r", loc=loc, ip=ip)


@dsl_user_op
def bulk_load(dst, src, nbytes, mbar, *, loc=None, ip=None):
    _asm([], [_u(dst, loc, ip), Int64(src).ir_value(loc=loc, ip=ip), _u(nbytes, loc, ip), _u(mbar, loc, ip)],
         "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
         "r,l,r,r", loc=loc, ip=ip)


@dsl_user_op
def tma_store_4d(tmap, c0, c1, c2, c3, src, *, loc=None, ip=None):
    _asm([], [Int64(tmap).ir_value(loc=loc, ip=ip)] + [Int32(c).ir_value(loc=loc, ip=ip) for c in (c0, c1, c2, c3)] + [_u(src, loc, ip)],
         "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [$0, {$1, $2, $3, $4}], [$5];",
         "l,r,r,r,r,r", loc=loc, ip=ip)


@dsl_user_op
def prefetch_l2_bulk(addr, nbytes, *, loc=None, ip=None):
    _asm([], [Int64(addr).ir_value(loc=loc, ip=ip), _u(nbytes, loc, ip)], "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r",
         loc=loc, ip=ip)


@dsl_user_op
def prefetch_tmap(addr, *, loc=None, ip=None):
    _asm([], [Int64(addr).ir_value(loc=loc, ip=ip)], "prefetch.tensormap [$0];", "l", loc=loc, ip=ip)


@dsl_user_op
def bulk_commit(*, loc=None, ip=None):
    _asm([], [], "cp.async.bulk.commit_group;", "", loc=loc, ip=ip)


@dsl_user_op
def bulk_wait_read0(*, loc=None, ip=None):
    _asm([], [], "cp.async.bulk.wait_group.read 0;", "", loc=loc, ip=ip)


@dsl_user_op
def bulk_wait0(*, loc=None, ip=None):
    _asm([], [], "cp.async.bulk.wait_group 0;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_init(addr, count: cutlass.Constexpr, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)], f"mbarrier.init.shared::cta.b64 [$0], {count};", "r", loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive(addr, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)], "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];", "r", loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive_tx(addr, nbytes: cutlass.Constexpr, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip)], f"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], {nbytes};", "r", loc=loc, ip=ip)


@dsl_user_op
def mbar_wait(addr, parity, *, loc=None, ip=None):
    _asm([], [_u(addr, loc, ip), _u(parity, loc, ip)],
         "{ .reg .pred P; WAIT_${:uid}: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [$0], $1; @!P bra WAIT_${:uid}; }",
         "r,r", loc=loc, ip=ip)


@dsl_user_op
def named_bar(bid, nthreads: cutlass.Constexpr, *, loc=None, ip=None):
    _asm([], [_u(bid, loc, ip)], f"bar.sync $0, {nthreads};", "r", loc=loc, ip=ip)


# ----------------------------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------------------------
@cute.jit
def tf32_split(x: Float32):
    """x = hi + lo with hi exact in tf32 (truncation); the MMA reads the top 19 bits of lo (|error| < 2^-21 |x|)."""
    hi = (x.bitcast(Uint32) & Uint32(0xFFFFE000)).bitcast(Float32)
    return hi, x - hi


@cute.jit
def tf32_split3(x: Float32):
    """x = hi + lo + lo2 exactly, each term representable in tf32 (truncation splits)."""
    hi = (x.bitcast(Uint32) & Uint32(0xFFFFE000)).bitcast(Float32)
    r = x - hi
    lo = (r.bitcast(Uint32) & Uint32(0xFFFFE000)).bitcast(Float32)
    return hi, lo, r - lo


@cute.jit
def half_split2(a: Float32, b: Float32):
    """fp16x2 high parts and fp16x2 low parts of (a, b)."""
    h = f2h2(a, b)
    ah, bh = h2f2(h)
    da, db = fsub2(a, b, ah, bh)
    l = f2h2(da, db)
    return h, l


@cute.jit
def sw32_row(base: Uint32, r):
    """Byte address of the two 16-byte halves of row r in a K/MN SW32 canonical tile (32-byte rows)."""
    rr = Uint32(r)
    a = base + (rr >> Uint32(3)) * Uint32(256) + (rr & Uint32(7)) * Uint32(32)
    sw = ((rr >> Uint32(2)) & Uint32(1)) * Uint32(16)
    return a + sw, a + (Uint32(16) - sw)


@cute.jit
def gram_partial_store(y: cute.Tensor, lane, gw, gbuf: Uint32, p: cutlass.Constexpr):
    """Form two-row Gram products, then a transposed warp reduction (11 shuffles).
    The permuted entries are stored as per-warp partials at gbuf + 160 p + 16 e + 4 w."""
    b4 = (lane >> 4) & 1
    b3 = (lane >> 3) & 1
    b2 = (lane >> 2) & 1
    b1 = (lane >> 1) & 1
    sl = b3 * 3 + b2 * 2 + b1
    ent = b4 * 5 + sl
    writer = ((lane & 1) == 0) & ((b3 == 0) | (b2 == 0)) & (sl < 5) & (((b3 * 4 + b2 * 2 + b1) != 3))
    # Splitting the ten entries into two sets related by the cycle 0->1->2->3->0
    # lets each lane form five two-row products, rather than all ten products plus selects.
    z = cute.make_rmem_tensor(4, Float32)
    a = cute.make_rmem_tensor(4, Float32)
    b = cute.make_rmem_tensor(4, Float32)
    for i in cutlass.range_constexpr(4):
        z[i] = shfl_bfly(y[i], 16)
    for i in cutlass.range_constexpr(4):
        a[i] = y[i]
        b[i] = z[i]
        if b4 == 1:
            a[i] = y[(i + 1) % 4]
            b[i] = z[(i + 1) % 4]
    r5 = cute.make_rmem_tensor(6, Float32)
    p0, p1 = fmul2(a[0], a[2], a[0], a[2])
    r5[0], r5[1] = ffma2(b[0], b[2], b[0], b[2], p0, p1)
    p0, p1 = fmul2(a[0], a[2], a[1], a[3])
    r5[2], r5[3] = ffma2(b[0], b[2], b[1], b[3], p0, p1)
    r5[4] = a[0] * a[2] + b[0] * b[2]
    r5[5] = Float32(0)
    r3 = cute.make_rmem_tensor(4, Float32)
    for i in cutlass.range_constexpr(3):
        send = r5[3 + i]
        keep = r5[i]
        if b3 == 1:
            send = r5[i]
            keep = r5[3 + i]
        r3[i] = keep + shfl_bfly(send, 8)
    r3[3] = Float32(0)
    r2 = cute.make_rmem_tensor(2, Float32)
    for i in cutlass.range_constexpr(2):
        send = r3[2 + i]
        keep = r3[i]
        if b2 == 1:
            send = r3[i]
            keep = r3[2 + i]
        r2[i] = keep + shfl_bfly(send, 4)
    send = r2[1]
    keep = r2[0]
    if b1 == 1:
        send = r2[0]
        keep = r2[1]
    r1 = keep + shfl_bfly(send, 2)
    r1 = r1 + shfl_bfly(r1, 1)
    if writer:
        sts32(gbuf + Uint32(p * 160) + Uint32(16) * ent.to(Uint32) + Uint32(4) * gw.to(Uint32), r1.bitcast(Uint32))


@cute.jit
def gram_chol_paired(gbuf: Uint32, p: cutlass.Constexpr, cf: cute.Tensor, lane):
    """Factor the summed Gram matrix with paired independent entries and the original pivot floors."""
    gg = cute.make_rmem_tensor(10, Float32)
    # lanes 0-9 each sum one entry's four per-warp partials; the ten sums are broadcast by shuffles
    v4 = lds128f(gbuf + Uint32(p * 160) + Uint32(16) * cutlass.min(Uint32(lane), Uint32(9)))
    tot = (v4[0] + v4[1]) + (v4[2] + v4[3])
    # gram_partial_store orders the entries as 00,22,01,23,02,11,33,12,03,13.
    for e in cutlass.range_constexpr(10):
        gg[e] = shfl_idx(tot, [0, 2, 5, 4, 7, 1, 8, 9, 3, 6][e])
    l0 = rsqrt_approx(cute.arch.fmax(gg[0], Float32(1e-30)))
    l10 = gg[1] * l0
    # The two independent rows share each preceding inverse diagonal.
    l20, l30 = fmul2(gg[3], gg[6], l0, l0)
    d1 = gg[2] - l10 * l10
    if cutlass.const_expr(p == 0):
        l1 = rsqrt_approx(cute.arch.fmax(cute.arch.fmax(d1, Float32(1e-6) * gg[2]), Float32(1e-30)))
    else:
        l1 = rsqrt_approx(cute.arch.fmax(d1, Float32(1e-30)))
        if d1 <= Float32(TAU2) * gg[2]:
            l1 = Float32(0)
    a21, a31 = ffma2(-l20, -l30, l10, l10, gg[4], gg[7])
    l21, l31 = fmul2(a21, a31, l1, l1)
    d2a, d3a = ffma2(-l20, -l30, l20, l30, gg[5], gg[9])
    d2, d3a = ffma2(-l21, -l31, l21, l31, d2a, d3a)
    if cutlass.const_expr(p == 0):
        l2 = rsqrt_approx(cute.arch.fmax(cute.arch.fmax(d2, Float32(1e-6) * gg[5]), Float32(1e-30)))
    else:
        l2 = rsqrt_approx(cute.arch.fmax(d2, Float32(1e-30)))
        if d2 <= Float32(TAU2) * gg[5]:
            l2 = Float32(0)
    l32 = (gg[8] - l30 * l20 - l31 * l21) * l2
    d3 = d3a - l32 * l32
    if cutlass.const_expr(p == 0):
        l3 = rsqrt_approx(cute.arch.fmax(cute.arch.fmax(d3, Float32(1e-6) * gg[9]), Float32(1e-30)))
    else:
        l3 = rsqrt_approx(cute.arch.fmax(d3, Float32(1e-30)))
        if d3 <= Float32(TAU2) * gg[9]:
            l3 = Float32(0)
    cf[0] = l0
    cf[1] = l10
    cf[2] = l1
    cf[3] = l20
    cf[4] = l21
    cf[5] = l2
    cf[6] = l30
    cf[7] = l31
    cf[8] = l32
    cf[9] = l3
    bad = Boolean(False)
    good = Boolean(False)
    if cutlass.const_expr(p == 0):
        bad = (gg[0] > Float32(1e-30)) & ((d1 < Float32(1e-3) * gg[2]) | (d2 < Float32(1e-3) * gg[5]) | (d3 < Float32(1e-3) * gg[9]))
        good = (gg[0] > Float32(1e-30)) & (d1 > Float32(TAU1) * gg[2]) & (d2 > Float32(TAU1) * gg[5]) & (d3 > Float32(TAU1) * gg[9])
    return bad, good


@cute.jit
def gram_chol_scalar(gbuf: Uint32, p: cutlass.Constexpr, cf: cute.Tensor, lane):
    """Factor the summed Gram matrix; for the first pass return (needs the precision path, one pass suffices)."""
    gg = cute.make_rmem_tensor(10, Float32)
    # lanes 0-9 each sum one entry's four per-warp partials; the ten sums are broadcast by shuffles
    v4 = lds128f(gbuf + Uint32(p * 160) + Uint32(16) * cutlass.min(Uint32(lane), Uint32(9)))
    tot = (v4[0] + v4[1]) + (v4[2] + v4[3])
    # gram_partial_store orders the entries as 00,22,01,23,02,11,33,12,03,13.
    for e in cutlass.range_constexpr(10):
        gg[e] = shfl_idx(tot, [0, 2, 5, 4, 7, 1, 8, 9, 3, 6][e])
    l0 = rsqrt_approx(cute.arch.fmax(gg[0], Float32(1e-30)))
    l10 = gg[1] * l0
    l1 = Float32(0)
    l2 = Float32(0)
    l3 = Float32(0)
    if cutlass.const_expr(p == 0):
        # first pass: a relative pivot floor damps numerically dependent columns (their direction is kept)
        l1 = rsqrt_approx(cute.arch.fmax(cute.arch.fmax(gg[2] - l10 * l10, Float32(1e-6) * gg[2]), Float32(1e-30)))
    else:
        # second pass: a column without an independent part (relative to its norm) is dropped, which keeps the
        # factor consistent with the vectors and the result orthonormal on the kept columns
        d1 = gg[2] - l10 * l10
        l1 = rsqrt_approx(cute.arch.fmax(d1, Float32(1e-30)))
        if d1 <= Float32(TAU2) * gg[2]:
            l1 = Float32(0)
    l20 = gg[3] * l0
    l21 = (gg[4] - l20 * l10) * l1
    if cutlass.const_expr(p == 0):
        l2 = rsqrt_approx(cute.arch.fmax(cute.arch.fmax(gg[5] - l20 * l20 - l21 * l21, Float32(1e-6) * gg[5]), Float32(1e-30)))
    else:
        d2 = gg[5] - l20 * l20 - l21 * l21
        l2 = rsqrt_approx(cute.arch.fmax(d2, Float32(1e-30)))
        if d2 <= Float32(TAU2) * gg[5]:
            l2 = Float32(0)
    l30 = gg[6] * l0
    l31 = (gg[7] - l30 * l10) * l1
    l32 = (gg[8] - l30 * l20 - l31 * l21) * l2
    if cutlass.const_expr(p == 0):
        l3 = rsqrt_approx(cute.arch.fmax(cute.arch.fmax(gg[9] - l30 * l30 - l31 * l31 - l32 * l32, Float32(1e-6) * gg[9]), Float32(1e-30)))
    else:
        d3 = gg[9] - l30 * l30 - l31 * l31 - l32 * l32
        l3 = rsqrt_approx(cute.arch.fmax(d3, Float32(1e-30)))
        if d3 <= Float32(TAU2) * gg[9]:
            l3 = Float32(0)
    cf[0] = l0
    cf[1] = l10
    cf[2] = l1
    cf[3] = l20
    cf[4] = l21
    cf[5] = l2
    cf[6] = l30
    cf[7] = l31
    cf[8] = l32
    cf[9] = l3
    bad = Boolean(False)
    good = Boolean(False)
    if cutlass.const_expr(p == 0):
        d1 = gg[2] - l10 * l10
        d2 = gg[5] - l20 * l20 - l21 * l21
        d3 = gg[9] - l30 * l30 - l31 * l31 - l32 * l32
        bad = (gg[0] > Float32(1e-30)) & ((d1 < Float32(1e-3) * gg[2]) | (d2 < Float32(1e-3) * gg[5]) | (d3 < Float32(1e-3) * gg[9]))
        good = (gg[0] > Float32(1e-30)) & (d1 > Float32(TAU1) * gg[2]) & (d2 > Float32(TAU1) * gg[5]) & (d3 > Float32(TAU1) * gg[9])
    return bad, good


@cute.jit
def gram_chol(gbuf: Uint32, p: cutlass.Constexpr, cf: cute.Tensor, lane, paired: cutlass.Constexpr):
    # Match the existing small/large launch choice. Both variants are compiled
    # during Pool construction; the flush call never compiles or reads due flags.
    if cutlass.const_expr(paired):
        result = gram_chol_paired(gbuf, p, cf, lane)
    else:
        result = gram_chol_scalar(gbuf, p, cf, lane)
    return result


@cute.jit
def mgs_rows(y: cute.Tensor, lane, gw, buf: Uint32, bar: Uint32):
    # One row per thread; alternate two partial-sum buffers so a following warp store cannot race a reader.
    for j in cutlass.range_constexpr(4):
        v = y[j]
        for rep in cutlass.range_constexpr(2):
            for c in cutlass.range_constexpr(j):
                dot = v * y[c]
                for bit in cutlass.range_constexpr(5):
                    dot = dot + shfl_bfly(dot, 16 >> bit)
                addr = buf + Uint32(32 * ((j * j + rep * j + c) % 2))
                if lane == 0:
                    sts32(addr + Uint32(4) * gw.to(Uint32), dot.bitcast(Uint32))
                named_bar(bar, 128)
                vv = lds128f(addr)
                dot = (vv[0] + vv[1]) + (vv[2] + vv[3])
                v = v - dot * y[c]
        norm2 = v * v
        for bit in cutlass.range_constexpr(5):
            norm2 = norm2 + shfl_bfly(norm2, 16 >> bit)
        addr = buf + Uint32(32 * ((j * j + 2 * j) % 2))
        if lane == 0:
            sts32(addr + Uint32(4) * gw.to(Uint32), norm2.bitcast(Uint32))
        named_bar(bar, 128)
        vv = lds128f(addr)
        norm2 = (vv[0] + vv[1]) + (vv[2] + vv[3])
        y[j] = v * rsqrt_approx(cute.arch.fmax(norm2, Float32(1e-30)))


@cute.jit
def write_state_part(st: Uint32, tmS: Uint32, lane_off: Uint32, t, tx7: Uint32,
                     tail: Uint32, low: cutlass.Constexpr):
    srow = st + (t.to(Uint32) >> Uint32(3)) * Uint32(1024) + tx7 * Uint32(128)
    for blk in cutlass.range_constexpr(8):
        if cutlass.const_expr(blk < 7):
            vv = tmem_wait_ld(tmem_ld16_nw(tmS + lane_off + Uint32(16 * blk)))
        else:
            vv = []
            for i in cutlass.range_constexpr(4):
                vv += lds128f(tail + Uint32(16) * t.to(Uint32) + Uint32(2048 * i))
        for q4 in cutlass.range_constexpr(2):
            ph = cute.make_rmem_tensor(4, Uint32)
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(low):
                    unused, ph[j] = half_split2(vv[8 * q4 + 2 * j], vv[8 * q4 + 2 * j + 1])
                else:
                    ph[j] = f2h2(vv[8 * q4 + 2 * j], vv[8 * q4 + 2 * j + 1])
            ch = Uint32((blk % 4) * 2 + q4)
            sts128(srow + Uint32((blk // 4) * 16384) + ((ch ^ tx7) * Uint32(16)), ph[0], ph[1], ph[2], ph[3])


@cute.jit
def tri_apply(cf: cute.Tensor, y: cute.Tensor):
    q0 = y[0] * cf[0]
    q1 = (y[1] - q0 * cf[1]) * cf[2]
    q2 = (y[2] - q0 * cf[3] - q1 * cf[4]) * cf[5]
    q3 = (y[3] - q0 * cf[6] - q1 * cf[7] - q2 * cf[8]) * cf[9]
    y[0] = q0
    y[1] = q1
    y[2] = q2
    y[3] = q3


# ----------------------------------------------------------------------------------------------
# the kernel
# ----------------------------------------------------------------------------------------------
@cute.kernel
def flush_kernel(raw: cute.Pointer, kbp: cute.Pointer, ubp: cute.Pointer, wp: cute.Pointer, pp: cute.Pointer,
                 hcnt: cute.Pointer, idx: cute.Pointer, q0p: cute.Pointer, n: Int32, ns: Int32, tmaps: cute.Pointer, early_exit: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nblk, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(tid // 32)
    lane = tid % 32

    smem = SmemAllocator()
    stages_raw = smem.allocate_array(Int8, NSTAGE * STAGE, byte_alignment=1024)
    au_raw = smem.allocate_array(Int8, NG * 8192, byte_alignment=1024)
    ao_raw = smem.allocate_array(Int8, NG * 8192, byte_alignment=1024)      # AO (4 KB) then BO (4 KB)
    qb_raw = smem.allocate_array(Int8, NG * 6144 + 2048, byte_alignment=1024)   # per group QB (4 KiB) | Gram/column partials (2 KiB), then compact Q0B (2 KiB)
    rs_raw = smem.allocate_array(Int8, NG * 512, byte_alignment=16)
    bd_raw = smem.allocate_array(Int8, NG * 128 + 64, byte_alignment=16)
    mb_raw = smem.allocate_array(Int64, 3 * NSTAGE + NG, byte_alignment=8)
    misc = smem.allocate_array(Int32, NSTAGE + 4, byte_alignment=16)

    stage0 = Uint32(stages_raw.toint())
    au0 = Uint32(au_raw.toint())
    ao0 = Uint32(ao_raw.toint())
    qb0 = Uint32(qb_raw.toint())
    ys0 = qb0 + Uint32(4096)
    cp0 = qb0 + Uint32(4096)
    rs0 = Uint32(rs_raw.toint())
    bd0 = Uint32(bd_raw.toint())
    mb0 = Uint32(mb_raw.toint())
    misc0 = Uint32(misc.toint())
    full0 = mb0
    empty0 = mb0 + Uint32(8 * NSTAGE)
    mma0 = mb0 + Uint32(16 * NSTAGE)
    code0 = mb0 + Uint32(16 * NSTAGE + 8 * NG)

    if tid == 0:
        for s in cutlass.range_constexpr(NSTAGE):
            mbar_init(full0 + Uint32(8 * s), 1)
            mbar_init(code0 + Uint32(8 * s), 1)
            mbar_init(empty0 + Uint32(8 * s), 1)
        for g in cutlass.range_constexpr(NG):
            mbar_init(mma0 + Uint32(8 * g), 1)
        for s in cutlass.range_constexpr(NSTAGE):
            sts32(misc0 + Uint32(4 * s), Uint32(0xFFFFFFFF))     # no program held by the stage yet
    if cutlass.const_expr(not early_exit):
        if warp == 0:
            tmem_alloc(misc0 + Uint32(4 * NSTAGE), 512)
    cute.arch.mbarrier_init_fence()

    # global byte addresses
    raw_b = Int64(raw.toint())
    kb_b = Int64(kbp.toint())
    ub_b = Int64(ubp.toint())
    w_b = Int64(wp.toint())
    p_b = Int64(pp.toint())
    hc_b = Int64(hcnt.toint())
    idx_b = Int64(idx.toint())
    tmap_codes = Int64(tmaps.toint())
    tmap_kb = tmap_codes + Int64(128)
    tmap_c32 = tmap_codes + Int64(256)
    tmap_ub = tmap_codes + Int64(384)

    # the producer's first batch of due checks is issued before the init barrier (their latency overlaps it)
    total = n * Int32(HV)
    due0 = Boolean(False)
    prog0 = Int32(0)
    if warp == PROD_WARP + 1:
        if lane < 4:
            prefetch_tmap(tmap_codes + Int64(128) * Int64(lane))   # codes, kbuf, codes32, ubuf descriptors (idle warp)
    if warp == PROD_WARP + 3:
        if lane == 0:
            prefetch_l2_bulk(Int64(q0p.toint()), Uint32(V * R * 4))          # q0 (2 KB), read by the consumer setup
    if warp == PROD_WARP + 2:
        # the due check's second (dependent) load reads the fill counters: bring the counter array into L2, one slice
        # per CTA, while the first loads are in flight
        hbytes = ns * Int32(HV * 4)
        if lane == 0:
            if (ns > Int32(0)) & (ns <= Int32((1 << 22) // (HV * 4))):   # counter arrays up to 4 MB
                chunk = ((hbytes + nblk - Int32(1)) // nblk + Int32(15)) & Int32(-16)
                start = Int32(bid) * chunk
                if start < hbytes:
                    prefetch_l2_bulk(hc_b + Int64(start), cutlass.min(chunk, hbytes - start).to(Uint32))
    if warp == PROD_WARP:
        c = Int32(bid) + Int32(lane) * nblk
        if c < total:
            nn = c // Int32(HV)
            hh = (c + Int32(21) * nn) & Int32(HV - 1)
            slot = ldg32(idx_b + Int64(nn) * Int64(4)).to(Int32)
            if slot > Int32(0):
                hc = ldg32_volatile(hc_b + (Int64(slot) * Int64(HV) + Int64(hh)) * Int64(4)).to(Int32)
                if hc == Int32(L):
                    due0 = Boolean(True)
                    prog0 = slot * Int32(HV) + hh

    seed_due = Boolean(False)
    if cutlass.const_expr(early_exit):
        if warp == 0:
            c = Int32(bid)
            if c < total:
                nn = c // Int32(HV)
                hh = (c + Int32(21) * nn) & Int32(HV - 1)
                slot = ldg32(idx_b + Int64(nn) * Int64(4)).to(Int32)
                if slot > Int32(0):
                    hc = ldg32_volatile(hc_b + (Int64(slot) * Int64(HV) + Int64(hh)) * Int64(4)).to(Int32)
                    seed_due = hc == Int32(L)
            if seed_due:
                tmem_alloc(misc0 + Uint32(4 * NSTAGE), 512)
        # A CTA that has no due heads need not initialize q0 or publish four
        # empty stages. Scan later batches only when the current mask is empty.
        first_mask = Uint32(0)
        first_cand = Int32(bid) + Int32(32) * nblk
        if warp == PROD_WARP:
            first_mask = Uint32(cute.arch.vote_ballot_sync(due0))
            while (first_mask == Uint32(0)) & (first_cand < total):
                c = first_cand + Int32(lane) * nblk
                due = Boolean(False)
                prog0 = Int32(0)
                if c < total:
                    nn = c // Int32(HV)
                    hh = (c + Int32(21) * nn) & Int32(HV - 1)
                    slot = ldg32(idx_b + Int64(nn) * Int64(4)).to(Int32)
                    if slot > Int32(0):
                        hc = ldg32_volatile(hc_b + (Int64(slot) * Int64(HV) + Int64(hh)) * Int64(4)).to(Int32)
                        if hc == Int32(L):
                            due = Boolean(True)
                            prog0 = slot * Int32(HV) + hh
                first_mask = Uint32(cute.arch.vote_ballot_sync(due))
                first_cand = first_cand + Int32(32) * nblk
            if lane == 0:
                sts32(misc0 + Uint32(4 * (NSTAGE + 1)), first_mask)

    cute.arch.sync_threads()                       # barriers initialized and initial scan published: producer can start
    tmem_base = Uint32(0)
    if cutlass.const_expr(not early_exit):
        tmem_base = lds32(misc0 + Uint32(4 * NSTAGE))

    has_work = Boolean(True)
    if cutlass.const_expr(early_exit):
        has_work = lds32(misc0 + Uint32(4 * (NSTAGE + 1))) != Uint32(0)
    if has_work:
        if warp >= PROD_WARP:
          cute.arch.setmaxregister_decrease(REG_PROD)
          if warp == PROD_WARP:
            # ------------------------------------------------------------------ producer
            if cutlass.const_expr(early_exit):
                mask = first_mask
                cand = first_cand
            else:
                mask = Uint32(cute.arch.vote_ballot_sync(due0))
                cand = Int32(bid) + Int32(32) * nblk
            prog = prog0
            k = Int32(0)
            ends = Int32(0)
            while ends < Int32(NG):
                # next due program: iterate over the cached due mask of a batch of 32 candidates (one per lane)
                pid = Int32(-1)
                looking = Boolean(True)
                while looking:
                    if mask == Uint32(0):
                        if cand >= total:
                            looking = Boolean(False)
                        else:
                            c = cand + Int32(lane) * nblk
                            due = Boolean(False)
                            prog = Int32(0)
                            if c < total:
                                nn = c // Int32(HV)
                                hh = (c + Int32(21) * nn) & Int32(HV - 1)
                                slot = ldg32(idx_b + Int64(nn) * Int64(4)).to(Int32)
                                if slot > Int32(0):
                                    hc = ldg32_volatile(hc_b + (Int64(slot) * Int64(HV) + Int64(hh)) * Int64(4)).to(Int32)
                                    if hc == Int32(L):
                                        due = Boolean(True)
                                        prog = slot * Int32(HV) + hh
                            mask = Uint32(cute.arch.vote_ballot_sync(due))
                            cand = cand + Int32(32) * nblk
                    if mask != Uint32(0):
                        first = cute.arch.popc((mask & (Uint32(0) - mask)) - Uint32(1)).to(Int32)
                        pid = cute.arch.shuffle_sync(prog, first)
                        mask = mask & (mask - Uint32(1))
                        looking = Boolean(False)
                s = k % Int32(NSTAGE)
                ph = (k // Int32(NSTAGE)) & Int32(1)
                mbar_wait(empty0 + Uint32(8) * s.to(Uint32), (ph ^ Int32(1)).to(Uint32))
                # window reset (w = 0, p = 1) of the program this stage held before: its inputs have been consumed
                old = lds32(misc0 + Uint32(4) * s.to(Uint32)).to(Int32)
                if old >= Int32(0):
                    if lane < 16:
                        stg32(w_b + Int64(old) * Int64(64) + Int64(4) * Int64(lane), Uint32(0))
                    if lane == 16:
                        stg32(p_b + Int64(old) * Int64(4), Float32(1.0).bitcast(Uint32))
                cute.arch.sync_warp()
                if lane == 0:
                    sts32(misc0 + Uint32(4) * s.to(Uint32), pid.to(Uint32))
                    fb = full0 + Uint32(8) * s.to(Uint32)
                    if pid >= Int32(0):
                        slot = pid // Int32(HV)
                        hh = pid - slot * Int32(HV)
                        st = stage0 + Uint32(STAGE) * s.to(Uint32)
                        mbar_arrive_tx(fb, LOAD_BYTES)
                        head_b = raw_b + (Int64(OFF) + Int64(slot) * Int64(SS) + Int64(hh) * Int64(16384)) * Int64(4)
                        bulk_load(st + Uint32(ST_SC), head_b + Int64(16384), Uint32(3072), fb)
                        bulk_load(st + Uint32(ST_W), w_b + Int64(pid) * Int64(64), Uint32(64), fb)
                        bulk_load(st + Uint32(ST_P), (p_b + Int64(pid) * Int64(4)) & Int64(-16), Uint32(16), fb)
                        tma_load_4d(st + Uint32(ST_UB), tmap_ub, Int32(0), Int32(0), Int32(0), pid, fb)
                        tma_load_4d(st + Uint32(ST_KB), tmap_kb, Int32(0), Int32(0), Int32(0), pid, fb)
                    else:
                        mbar_arrive(fb)
                # steady state: the 16 KB code tile (needed ~1.3K cycles after the small inputs) follows after a short
                # pause, which reduces the refill's interference with the other groups' phases
                if k >= Int32(NSTAGE):
                    nanosleep_ns(200)
                if lane == 0:
                    if pid >= Int32(0):
                        slot = pid // Int32(HV)
                        hh = pid - slot * Int32(HV)
                        cb = code0 + Uint32(8) * s.to(Uint32)
                        mbar_arrive_tx(cb, 16384)
                        tma_load_4d(stage0 + Uint32(STAGE) * s.to(Uint32) + Uint32(ST_CODES), tmap_codes, Int32(0), Int32(0), hh, slot, cb)
                if pid < Int32(0):
                    ends = ends + Int32(1)
                k = k + Int32(1)
        else:
            cute.arch.setmaxregister_increase(REG_CONS)
            if cutlass.const_expr(early_exit):
                if warp == 0:
                    if not seed_due:
                        tmem_alloc(misc0 + Uint32(4 * NSTAGE), 512)
            # ------------------------------------------------------------------ consumers
            # Q0 operand (MN-major INTER [K=128 v x N=8]: fp16 hi of q0 | fp16 lo of q0).
            # Normalize each column by its own power of two to preserve small columns of the starting subspace.
            # Consumer warps set it up while the producer streams the first stages.
            q0f = cute.recast_ptr(q0p, dtype=Float32)
            qv = cute.make_rmem_tensor(4, Float32)
            for c in cutlass.range_constexpr(4):
                qv[c] = Float32(0)
            if warp < 4:
                qmb = cute.make_rmem_tensor(4, Uint32)
                for c in cutlass.range_constexpr(4):
                    qv[c] = q0f[tid * 4 + c]
                for c in cutlass.range_constexpr(4):
                    qmb[c] = redux_max(qv[c].bitcast(Uint32) & Uint32(0x7FFFFFFF))
                if lane == 0:
                    sts128(bd0 + Uint32(NG * 128 + 16 * warp), qmb[0], qmb[1], qmb[2], qmb[3])
            named_bar(Uint32(15), 128 * NG)
            if warp < 4:
                # Lanes 0..3 reduce one column each, and broadcast their power-of-two scales.
                qc = Uint32(lane & 3)
                qmax = Uint32(0)
                for w4 in cutlass.range_constexpr(4):
                    qmax = cutlass.max(qmax, lds32(bd0 + Uint32(NG * 128 + 16 * w4) + Uint32(4) * qc))
                qe = ((qmax >> Uint32(23)) & Uint32(255)).to(Int32)
                qscale = Float32(1.0)
                if qmax != Uint32(0):
                    qscale = exp2_int(cutlass.min(cutlass.max(Int32(126) - qe, Int32(-126)), Int32(127)))
                for c in cutlass.range_constexpr(4):
                    qv[c] = qv[c] * shfl_idx(qscale, c)
                h01, l01 = half_split2(qv[0], qv[1])
                h23, l23 = half_split2(qv[2], qv[3])
                sts128(qb0 + Uint32(NG * 6144) + Uint32(16) * Uint32(tid), h01, h23, l01, l23)
            if warp < 4 * NG:
                a, b = sw32_row(qb0 + Uint32(6144) * Uint32(warp // 4), tid % 128)
                sts128(b, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                a, b = sw32_row(stage0 + Uint32(STAGE) * Uint32(warp // 4) + Uint32(28672), tid % 128)
                sts128(b, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            fence_async_smem()
            named_bar(Uint32(15), 128 * NG)
            if cutlass.const_expr(early_exit):
                tmem_base = lds32(misc0 + Uint32(4 * NSTAGE))
            g = warp // 4
            gw = warp % 4
            t = tid % 128
            bar = Uint32(1) + g.to(Uint32)
            leader = gw == g
            tmS = tmem_base + Uint32(128) * g.to(Uint32)
            tmY = tmS + Uint32(112)
            lane_off = (Uint32(32) * gw.to(Uint32)) << Uint32(16)
            mbar_g = mma0 + Uint32(8) * g.to(Uint32)
            au = au0 + Uint32(8192) * g.to(Uint32)
            ao = ao0 + Uint32(8192) * g.to(Uint32)
            bo = ao + Uint32(4096)
            qb = qb0 + Uint32(6144) * g.to(Uint32)
            q0b = qb0 + Uint32(6144 * NG)
            ys = ys0 + Uint32(6144) * g.to(Uint32)
            cpg = cp0 + Uint32(6144) * g.to(Uint32)
            da = qb                                       # scale-product A in QB; B uses the tail of the current input stage
            rsg = rs0 + Uint32(512) * g.to(Uint32)
            bdg = bd0 + Uint32(128) * g.to(Uint32)

            mph = Uint32(0)
            s = g                                         # stage s == group g (NSTAGE == NG)
            fph = Uint32(0)
            mbar_wait(full0 + Uint32(8) * s.to(Uint32), fph)
            pid = lds32(misc0 + Uint32(4) * s.to(Uint32)).to(Int32)
            while pid >= Int32(0):
                st = stage0 + Uint32(STAGE) * s.to(Uint32)
                db = st + Uint32(28672)                 # 4 KiB unused by TMA inputs, consumed before S overwrites the stage
                slot = pid // Int32(HV)
                hh = pid - slot * Int32(HV)
                head_b = raw_b + (Int64(OFF) + Int64(slot) * Int64(SS) + Int64(hh) * Int64(16384)) * Int64(4)
                # ---------------------------------------------------------- P1: inputs and the power-of-two scale
                pval = lds32f(st + Uint32(ST_P) + Uint32(4) * (pid & Int32(3)).to(Uint32))
                skt = lds32f(st + Uint32(ST_SC) + Uint32(4) * t.to(Uint32))
                svt = lds32f(st + Uint32(ST_SC + 512) + Uint32(4) * t.to(Uint32))
                uo = cute.make_rmem_tensor(4, Float32)
                qo = cute.make_rmem_tensor(4, Float32)
                mq = Float32(0)
                mu = Float32(0)
                for r in cutlass.range_constexpr(4):
                    uo[r] = h2f(lds16(st + Uint32(ST_SC + 1024 + 512 * r) + Uint32(2) * t.to(Uint32)))
                    qo[r] = h2f(lds16(st + Uint32(ST_SC + 1024 + 512 * r + 256) + Uint32(2) * t.to(Uint32)))
                    mu = cute.arch.fmax(mu, cute.math.abs(uo[r]))
                    mq = cute.arch.fmax(mq, cute.math.abs(qo[r]))
                # thread t covers the 16 values at byte 16 t of both 64-column halves of the (SW128) kbuf and ubuf tiles:
                # row j = t // 8 of each; the B operand of the update product is 2^sig w_j kbuf_j (ubuf is A as loaded)
                wj = lds32f(st + Uint32(ST_W) + Uint32(4) * (t.to(Uint32) >> Uint32(3)))
                kw = lds128(st + Uint32(ST_KB) + Uint32(16) * t.to(Uint32))
                kw2 = lds128(st + Uint32(ST_KB + 2048) + Uint32(16) * t.to(Uint32))
                uw = lds128(st + Uint32(ST_UB) + Uint32(16) * t.to(Uint32))
                uw2 = lds128(st + Uint32(ST_UB + 2048) + Uint32(16) * t.to(Uint32))
                kx = [bmaxabs2(kw[e], kw2[e]) for e in range(4)]
                kx = bmaxabs2(bmaxabs2(kx[0], kx[1]), bmaxabs2(kx[2], kx[3]))
                mkb = (cute.math.abs(wj) * cutlass.max((kx << Uint32(16)) & Uint32(0x7FFF0000), kx & Uint32(0x7FFF0000)).bitcast(Float32)).bitcast(Uint32)
                ux = [bmaxabs2(uw[e], uw2[e]) for e in range(4)]
                ux = bmaxabs2(bmaxabs2(ux[0], ux[1]), bmaxabs2(ux[2], ux[3]))
                mub = cutlass.max((ux << Uint32(16)) & Uint32(0x7FFF0000), ux & Uint32(0x7FFF0000))
                r0 = redux_max(skt.bitcast(Uint32) & Uint32(0x7FFFFFFF))
                r1 = redux_max(svt.bitcast(Uint32) & Uint32(0x7FFFFFFF))
                r2 = redux_max(mq.bitcast(Uint32))
                r3 = redux_max(mu.bitcast(Uint32))
                r4 = redux_max(mub)
                r5 = redux_max(mkb)
                # scale product D[v][k] = (p s_v[v] / 127) s_k[k] by one tf32 MMA: (bh + bl)(ah + al) from two-term
                # truncation splits (relative error < 2^-19); A's K slots 4-7 stay zero (written once), B's hold finite data
                bh, bl = tf32_split((pval * svt) * Float32(256.0 / 127.0))
                ah, al = tf32_split(skt)
                a, b = sw32_row(da, t)
                sts128(a, bh.bitcast(Uint32), bh.bitcast(Uint32), bl.bitcast(Uint32), bl.bitcast(Uint32))
                a, b = sw32_row(db, t)
                sts128(a, ah.bitcast(Uint32), al.bitcast(Uint32), ah.bitcast(Uint32), al.bitcast(Uint32))
                tc_before()
                if lane == 0:
                    bulk_wait_read0()                               # previous program's code tiles (AU/AO) read
                    b = bdg + Uint32(32) * gw.to(Uint32)
                    sts128(b, r0, r1, r2, r3)
                    sts32(b + Uint32(16), r4)
                    sts32(b + Uint32(20), r5)
                named_bar(bar, 128)
                if leader:
                    # The barrier acquires every writer's stores; this fence publishes them to the async reader.
                    fence_async_smem()
                    tc_after()
                    with cute.arch.elect_one():
                        mma_list(tmS, mbar_g, [da >> Uint32(4), db >> Uint32(4)], [("tf32", 0, "0xC000401000010000", 1, "0xC000401000010000", "0x08200910", False)])
                # Six lanes reduce one bound component each; broadcast the six maxima to the warp.
                boff = Uint32(4) * cutlass.min(Uint32(lane), Uint32(5))
                mm = lds32(bdg + boff)
                for w4 in cutlass.range_constexpr(1, 4):
                    mm = cutlass.max(mm, lds32(bdg + Uint32(32 * w4) + boff))
                m0 = shfl_idx(mm.bitcast(Float32), 0).bitcast(Uint32)
                m1 = shfl_idx(mm.bitcast(Float32), 1).bitcast(Uint32)
                m2 = shfl_idx(mm.bitcast(Float32), 2).bitcast(Uint32)
                m3 = shfl_idx(mm.bitcast(Float32), 3).bitcast(Uint32)
                m4 = shfl_idx(mm.bitcast(Float32), 4).bitcast(Uint32)
                m5 = shfl_idx(mm.bitcast(Float32), 5).bitcast(Uint32)
                bound = pval * (m0.bitcast(Float32) * m1.bitcast(Float32) + Float32(4.0) * m2.bitcast(Float32) * m3.bitcast(Float32)) \
                    + Float32(16.0) * m4.bitcast(Float32) * m5.bitcast(Float32)
                be = ((bound.bitcast(Uint32) >> Uint32(23)) & Uint32(255)).to(Int32) - Int32(127)   # bound < 2^(be+1)
                sig = Int32(0)
                if bound > Float32(0):
                    sig = (Int32(6) - be) & Int32(-2)           # bound 2^sig < 2^8: |S^T q0| <= 128 2^8 fits fp16
                    sig = cutlass.min(cutlass.max(sig, Int32(-120)), Int32(104))
                sh = sig // Int32(2)
                sc_h = exp2_int(sh)                     # 2^(sig/2)
                sc = sc_h * sc_h                        # 2^sig
                isc_h = exp2_int(-sh)
                isc = isc_h * isc_h                     # 2^-sig (normal: sig in [-120, 104])
                # ---------------------------------------------------------- P1b: MMA operands
                # KW (in AU): hi/lo bf16 of 2^sig w_j kbuf[j][k], same SW128 MN-major layout as the kbuf tile
                hv = cute.make_rmem_tensor(8, Uint32)
                lv = cute.make_rmem_tensor(8, Uint32)
                fw = wj * sc
                kws = [kw[0], kw[1], kw[2], kw[3], kw2[0], kw2[1], kw2[2], kw2[3]]
                for e in cutlass.range_constexpr(8):
                    k0, k1 = bf2f2(kws[e])
                    x0, x1 = fmul2(k0, k1, fw, fw)
                    hv[e] = prmt(x0.bitcast(Uint32), x1.bitcast(Uint32), Uint32(0x7632))   # truncated bf16 high parts
                    n0, n1 = bfsub2(hv[e], x0, x1)                                        # -(low parts), exact
                    lv[e] = f2bf2(n0, n1)                     # bf16 of -lo: the second update MMA negates B
                sts128(au + Uint32(16) * t.to(Uint32), hv[0], hv[1], hv[2], hv[3])
                sts128(au + Uint32(2048) + Uint32(16) * t.to(Uint32), hv[4], hv[5], hv[6], hv[7])
                sts128(au + Uint32(4096) + Uint32(16) * t.to(Uint32), lv[0], lv[1], lv[2], lv[3])
                sts128(au + Uint32(6144) + Uint32(16) * t.to(Uint32), lv[4], lv[5], lv[6], lv[7])
                # AO row t: tf32 hi/lo of 2^sig p q_old[r][t];  BO row t: u_old[r][t] twice
                psc = pval * sc
                th = cute.make_rmem_tensor(4, Float32)
                tl = cute.make_rmem_tensor(4, Float32)
                for r in cutlass.range_constexpr(4):
                    th[r], tl[r] = tf32_split(psc * qo[r])
                a, b = sw32_row(ao, t)
                sts128(a, th[0].bitcast(Uint32), th[1].bitcast(Uint32), th[2].bitcast(Uint32), th[3].bitcast(Uint32))
                sts128(b, tl[0].bitcast(Uint32), tl[1].bitcast(Uint32), tl[2].bitcast(Uint32), tl[3].bitcast(Uint32))
                a, b = sw32_row(bo, t)
                sts128(a, uo[0].bitcast(Uint32), uo[1].bitcast(Uint32), uo[2].bitcast(Uint32), uo[3].bitcast(Uint32))
                sts128(b, uo[0].bitcast(Uint32), uo[1].bitcast(Uint32), uo[2].bitcast(Uint32), uo[3].bitcast(Uint32))
                # ---------------------------------------------------------- P1c: dequantized residual -> TMEM (one code chunk ahead)
                # code c -> float 2^(23+sig) + (c + 128) 2^sig, minus (2^23 + 128) 2^sig = c 2^sig exactly; times D
                # the scale product in TMEM is 2^8 D: codes carry 2^(sig - 8)
                s8 = sig - Int32(8)
                s8c = cutlass.min(cutlass.max(s8, Int32(-24)), Int32(4))
                cbase = st + Uint32(ST_CODES) + Uint32(128) * t.to(Uint32)
                tx7 = Uint32(t % 8)
                mbar_wait(code0 + Uint32(8) * s.to(Uint32), fph)
                cks0 = [lds128(cbase + tx7 * Uint32(16)), lds128(cbase + ((Uint32(1) ^ tx7) * Uint32(16)))]
                mbar_wait(mbar_g, mph)
                mph = mph ^ Uint32(1)
                tc_after()
                if s8 != s8c:
                    # rare (|S| far from the usual range): rescale the scale product exactly by 2^(s8 - s8c) so that
                    # codes scaled by 2^s8c still give 2^sig c D
                    fix = exp2_int(s8 - s8c)
                    for blk in cutlass.range_constexpr(8):
                        vv = tmem_wait_ld(tmem_ld16_nw(tmS + lane_off + Uint32(16 * blk)))
                        for i in cutlass.range_constexpr(8):
                            vv[2 * i], vv[2 * i + 1] = fmul2(vv[2 * i], vv[2 * i + 1], fix, fix)
                        tmem_st16_nw(tmS + lane_off + Uint32(16 * blk), vv)
                    tmem_wait_st()
                dsc0 = tmem_wait_ld(tmem_ld16_nw(tmS + lane_off))

                mgw = ((Int32(25) + s8c).to(Uint32) << Uint32(2)) * Uint32(0x01010101)   # fp16 exponent byte of 2^s8 1024
                na = Float32(-1152.0) * exp2_int(s8c)
                dsc = dsc0
                cks = [cks0[0], cks0[1]]
                vprev = []
                for blk in cutlass.range_constexpr(8):
                    if cutlass.const_expr(blk < 7):
                        dn = tmem_ld16_nw(tmS + lane_off + Uint32(16 * (blk + 1)))
                    if cutlass.const_expr(blk < 6):
                        cks.append(lds128(cbase + ((Uint32(blk + 2) ^ tx7) * Uint32(16))))
                    cw = cks[blk]
                    vals = []
                    for w in cutlass.range_constexpr(4):
                        wx = cw[w] ^ Uint32(0x80808080)
                        c0, c1 = fhadd2(prmt(wx, mgw, Uint32(0x4140)), na)
                        c2, c3 = fhadd2(prmt(wx, mgw, Uint32(0x4342)), na)
                        c0, c1 = fmul2(c0, c1, dsc[4 * w], dsc[4 * w + 1])
                        c2, c3 = fmul2(c2, c3, dsc[4 * w + 2], dsc[4 * w + 3])
                        vals += [c0, c1, c2, c3]
                    if cutlass.const_expr(blk % 2 == 1):
                        tmem_st32_nw(tmS + lane_off + Uint32(16 * (blk - 1)), vprev + vals)   # two blocks per store
                    else:
                        vprev = vals
                    if cutlass.const_expr(blk < 7):
                        dsc = tmem_wait_ld(dn)
                tmem_wait_st()
                # ---------------------------------------------------------- build MMA
                tc_before()
                named_bar(bar, 128)
                if leader:
                    fence_async_smem()
                    tc_after()
                    with cute.arch.elect_one():
                        mma_list(tmS, mbar_g, [(st + Uint32(ST_UB)) >> Uint32(4), au >> Uint32(4), (au + Uint32(4096)) >> Uint32(4),
                                               ao >> Uint32(4), bo >> Uint32(4)],
                                 [("f16", 0, "0x4000404000800000", 1, "0x4000404000800000", "0x08218490", True), ("f16", 0, "0x4000404000800000", 2, "0x4000404000800000", "0x0821C490", True),
                                  ("tf32", 3, "0xC000401000010000", 4, "0xC000401000010000", "0x08200910", True)])
                mbar_wait(mbar_g, mph)
                mph = mph ^ Uint32(1)
                tc_after()
                tailS = cute.make_rmem_tensor(8, Float32)
                # ---------------------------------------------------------- P2: fp16 copy of S (SW128 K-major canonical in the stage)
                srow = st + (t.to(Uint32) >> Uint32(3)) * Uint32(1024) + tx7 * Uint32(128)
                v32 = tmem_wait_ld(tmem_ld32_nw(tmS + lane_off))
                for bp in cutlass.range_constexpr(4):                 # 32 columns per TMEM load
                    if cutlass.const_expr(bp < 3):
                        nxt = tmem_ld32_nw(tmS + lane_off + Uint32(32 * (bp + 1)))
                    if cutlass.const_expr(bp == 3):
                        for i in cutlass.range_constexpr(8):
                            tailS[i] = v32[16 + i]
                    for hb in cutlass.range_constexpr(2):
                        blk = 2 * bp + hb
                        for q4 in cutlass.range_constexpr(2):
                            ch = Uint32((blk % 4) * 2 + q4)
                            o = 16 * hb + q4 * 8
                            h0 = f2h2(v32[o + 0], v32[o + 1])
                            h1 = f2h2(v32[o + 2], v32[o + 3])
                            h2 = f2h2(v32[o + 4], v32[o + 5])
                            h3 = f2h2(v32[o + 6], v32[o + 7])
                            sts128(srow + Uint32((blk // 4) * 16384) + ((ch ^ tx7) * Uint32(16)), h0, h1, h2, h3)
                    if cutlass.const_expr(bp < 3):
                        v32 = tmem_wait_ld(nxt)
                tc_before()
                named_bar(bar, 128)
                if leader:
                    fence_async_smem()
                    tc_after()
                    with cute.arch.elect_one():
                        mma8_f16(tmY, st >> Uint32(4), q0b >> Uint32(4), mbar_g, "0x4000404004000000", [0, 128, 256, 384, 512, 640, 768, 896], "0x0000400000080000", [0, 16, 32, 48, 64, 80, 96, 112], "0x08038010")
                mbar_wait(mbar_g, mph)
                mph = mph ^ Uint32(1)
                tc_after()
                # ---------------------------------------------------------- P3: S^T q0 (scaled into fp16 range) -> B operand of S @ .
                # qv = orth(S S^T q0): the intermediate orthonormalization does not change the span (as in the baseline kernel)
                yv = tmem_ld8(tmY + lane_off)
                y4 = cute.make_rmem_tensor(4, Float32)
                for rp in cutlass.range_constexpr(2):
                    y4[2 * rp], y4[2 * rp + 1] = fadd2(yv[2 * rp], yv[2 * rp + 1], yv[2 * rp + 4], yv[2 * rp + 5])
                a, b = sw32_row(qb, t)
                sts64(a, f2h2(y4[0], y4[1]), f2h2(y4[2], y4[3]))      # B columns 4-7 (never read) keep stale bytes
                tc_before()
                named_bar(bar, 128)
                if leader:
                    fence_async_smem()
                    tc_after()
                    with cute.arch.elect_one():
                        mma8_f16(tmY, st >> Uint32(4), qb >> Uint32(4), mbar_g, "0x4000404000010000", [0, 2, 4, 6, 1024, 1026, 1028, 1030], "0xC000401000000000", [0, 32, 64, 96, 128, 160, 192, 224], "0x08030010")
                mbar_wait(mbar_g, mph)
                mph = mph ^ Uint32(1)
                tc_after()
                # ---------------------------------------------------------- P4: CholeskyQR2 of Y2 = S S^T q0 (one row per thread); W = S^T Q1 overlaps pass 2
                y2v = tmem_ld4(tmY + lane_off)                  # Y1 has no low part: columns 4-7 of S Y1 are zero
                y4 = cute.make_rmem_tensor(4, Float32)
                for r in cutlass.range_constexpr(4):
                    y4[r] = y2v[r]
                tc_before()
                gram_partial_store(y4, lane, gw, ys, 0)
                named_bar(bar, 128)
                cf1 = cute.make_rmem_tensor(10, Float32)
                bad, good = gram_chol(ys, 0, cf1, lane, early_exit)
                if not bad:
                    tri_apply(cf1, y4)
                    qh01 = f2h2(y4[0], y4[1])
                    qh23 = f2h2(y4[2], y4[3])
                    a, b = sw32_row(qb, t)
                    sts64(a, qh01, qh23)
                    if not good:
                        gram_partial_store(y4, lane, gw, ys, 1)
                    named_bar(bar, 128)
                    if leader:
                        fence_async_smem()
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), qb >> Uint32(4), mbar_g, "0x4000404004000000", [0, 128, 256, 384, 512, 640, 768, 896], "0xC000401000000000", [0, 32, 64, 96, 128, 160, 192, 224], "0x08038010")
                    # a well-conditioned first pass is already orthonormal far below the fp16 rounding of q: R2 = I
                    cf2 = cute.make_rmem_tensor(10, Float32)
                    for i in cutlass.range_constexpr(10):
                        cf2[i] = Float32(1.0) if i in (0, 2, 5, 9) else Float32(0.0)
                    if not good:
                        unused_b, unused_g = gram_chol(ys, 1, cf2, lane, early_exit)
                        tri_apply(cf2, y4)
                        qh01 = f2h2(y4[0], y4[1])
                        qh23 = f2h2(y4[2], y4[3])
                    # half 0: q_new row -> fp16, A of the E product
                    qnf = cute.make_rmem_tensor(4, Float32)
                    qnf[0], qnf[1] = h2f2(qh01)
                    qnf[2], qnf[3] = h2f2(qh23)
                    qrow = head_b + Int64(17408 + 256) + Int64(2) * Int64(t)
                    stg16(qrow, qh01)
                    stg16_hi(qrow + Int64(512), qh01)
                    stg16(qrow + Int64(1024), qh23)
                    stg16_hi(qrow + Int64(1536), qh23)
                    a, b = sw32_row(ao, t)
                    sts128(a, qnf[0].bitcast(Uint32), qnf[1].bitcast(Uint32), qnf[2].bitcast(Uint32), qnf[3].bitcast(Uint32))
                    sts128(b, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    # ---------------------------------------------------------- P5: u = W R^-1, E = S - q^T u (in TMEM)
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    if t == 0:
                        mbar_arrive(empty0 + Uint32(8) * s.to(Uint32))     # the fp16 copy of S has been read by all products

                    wv = tmem_ld4(tmY + lane_off)                  # Q1 has no low part: columns 4-7 of W are zero
                    for r in cutlass.range_constexpr(4):
                        y4[r] = wv[r]
                    tmem_st8_nw(tmS + lane_off + Uint32(112), [tailS[i] for i in range(8)])
                    tmem_wait_st()
                    if not good:
                        tri_apply(cf2, y4)
                    un = cute.make_rmem_tensor(4, Float32)
                    urow = head_b + Int64(17408) + Int64(2) * Int64(t)
                    for rp in cutlass.range_constexpr(2):
                        ua, ub = fmul2(y4[2 * rp], y4[2 * rp + 1], isc, isc)
                        uh = f2h2(ua, ub)
                        stg16(urow + Int64(1024 * rp), uh)
                        stg16_hi(urow + Int64(1024 * rp + 512), uh)
                        ua, ub = h2f2(uh)
                        un[2 * rp], un[2 * rp + 1] = fmul2(ua, ub, -sc, -sc)
                    a, b = sw32_row(bo, t)
                    sts128(a, un[0].bitcast(Uint32), un[1].bitcast(Uint32), un[2].bitcast(Uint32), un[3].bitcast(Uint32))
                else:
                    # Recompute the high-state projection, correct it with S_lo^T q0, then redo QR_k.
                    named_bar(bar, 128)
                    tl8 = tmem_ld8(tmS + lane_off + Uint32(120))
                    tfull = [tailS[i] for i in range(8)] + tl8
                    for i in cutlass.range_constexpr(4):
                        sts128(au + Uint32(16) * t.to(Uint32) + Uint32(2048 * i),
                               tfull[4 * i].bitcast(Uint32), tfull[4 * i + 1].bitcast(Uint32),
                               tfull[4 * i + 2].bitcast(Uint32), tfull[4 * i + 3].bitcast(Uint32))
                    # S_hi^T q0 again from the shared copy (still S_hi), then an empty commit keeps the number of
                    # MMA-barrier phases of this path even
                    tc_before()
                    named_bar(bar, 128)
                    if leader:
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), q0b >> Uint32(4), mbar_g, "0x4000404004000000", [0, 128, 256, 384, 512, 640, 768, 896], "0x0000400000080000", [0, 16, 32, 48, 64, 80, 96, 112], "0x08038010")
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    if leader:
                        with cute.arch.elect_one():
                            tcgen05.commit(cute.make_ptr(Uint64, mbar_g, cute.AddressSpace.smem))
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    write_state_part(st, tmS, lane_off, t, tx7, au, True)
                    tc_before()
                    named_bar(bar, 128)
                    if leader:
                        fence_async_smem()
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), q0b >> Uint32(4), mbar_g, "0x4000404004000000", [0, 128, 256, 384, 512, 640, 768, 896], "0x0000400000080000", [0, 16, 32, 48, 64, 80, 96, 112], "0x08038010", True)
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    yv = tmem_ld8(tmY + lane_off)
                    for rp in cutlass.range_constexpr(2):
                        ya, yb = fadd2(yv[2 * rp], yv[2 * rp + 1], yv[2 * rp + 4], yv[2 * rp + 5])
                        y4[2 * rp] = ya
                        y4[2 * rp + 1] = yb
                    mgs_rows(y4, lane, gw, ys, bar)
                    h01, l01 = half_split2(y4[0], y4[1])
                    h23, l23 = half_split2(y4[2], y4[3])
                    a, b = sw32_row(qb, t)
                    sts128(a, h01, h23, l01, l23)
                    tc_before()
                    named_bar(bar, 128)
                    if leader:
                        fence_async_smem()
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), qb >> Uint32(4), mbar_g, "0x4000404000010000", [0, 2, 4, 6, 1024, 1026, 1028, 1030], "0xC000401000000000", [0, 32, 64, 96, 128, 160, 192, 224], "0x08030010")
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    # S_lo @ Qk is in Y; restore S_hi and accumulate its contribution into Y.
                    write_state_part(st, tmS, lane_off, t, tx7, au, False)
                    tc_before()
                    named_bar(bar, 128)
                    if leader:
                        fence_async_smem()
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), qb >> Uint32(4), mbar_g, "0x4000404000010000", [0, 2, 4, 6, 1024, 1026, 1028, 1030], "0xC000401000000000", [0, 32, 64, 96, 128, 160, 192, 224], "0x08030010", True)
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    yv = tmem_ld8(tmY + lane_off)
                    for rp in cutlass.range_constexpr(2):
                        ya, yb = fadd2(yv[2 * rp], yv[2 * rp + 1], yv[2 * rp + 4], yv[2 * rp + 5])
                        y4[2 * rp] = ya
                        y4[2 * rp + 1] = yb
                    mgs_rows(y4, lane, gw, ys, bar)
                    h01, l01 = half_split2(y4[0], y4[1])
                    h23, l23 = half_split2(y4[2], y4[3])
                    a, b = sw32_row(qb, t)
                    sts128(a, h01, h23, l01, l23)
                    gram_partial_store(y4, lane, gw, ys, 1)
                    named_bar(bar, 128)
                    if leader:
                        fence_async_smem()
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), qb >> Uint32(4), mbar_g, "0x4000404004000000", [0, 128, 256, 384, 512, 640, 768, 896], "0xC000401000000000", [0, 32, 64, 96, 128, 160, 192, 224], "0x08038010")
                    cf2 = cute.make_rmem_tensor(10, Float32)
                    unused_b, unused_g = gram_chol(ys, 1, cf2, lane, early_exit)
                    tri_apply(cf2, y4)
                    # half 0: q_new row -> fp16, A of the E product
                    qnf = cute.make_rmem_tensor(4, Float32)
                    for r in cutlass.range_constexpr(4):
                        qh = f2h(y4[r])
                        qnf[r] = h2f(qh)
                        stg16(head_b + Int64(17408 + 512 * r + 256) + Int64(2) * Int64(t), qh)
                    a, b = sw32_row(ao, t)
                    sts128(a, qnf[0].bitcast(Uint32), qnf[1].bitcast(Uint32), qnf[2].bitcast(Uint32), qnf[3].bitcast(Uint32))
                    sts128(b, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    # ---------------------------------------------------------- P5: u = W R^-1, E = S - q^T u (in TMEM)
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    # W_hi = S_hi^T Q1 has finished. Accumulate W_lo before releasing the input stage.
                    write_state_part(st, tmS, lane_off, t, tx7, au, True)
                    tc_before()
                    named_bar(bar, 128)
                    if leader:
                        fence_async_smem()
                        tc_after()
                        with cute.arch.elect_one():
                            mma8_f16(tmY, st >> Uint32(4), qb >> Uint32(4), mbar_g, "0x4000404004000000", [0, 128, 256, 384, 512, 640, 768, 896], "0xC000401000000000", [0, 32, 64, 96, 128, 160, 192, 224], "0x08038010", True)
                    mbar_wait(mbar_g, mph)
                    mph = mph ^ Uint32(1)
                    tc_after()
                    if t == 0:
                        mbar_arrive(empty0 + Uint32(8) * s.to(Uint32))     # the fp16 copy of S has been read by all products

                    yv = tmem_ld8(tmY + lane_off)
                    for rp in cutlass.range_constexpr(2):
                        ya, yb = fadd2(yv[2 * rp], yv[2 * rp + 1], yv[2 * rp + 4], yv[2 * rp + 5])
                        y4[2 * rp] = ya
                        y4[2 * rp + 1] = yb
                    tail_restore = []
                    for i in cutlass.range_constexpr(4):
                        tail_restore += lds128f(au + Uint32(16) * t.to(Uint32) + Uint32(2048 * i))
                    tmem_st16_nw(tmS + lane_off + Uint32(112), tail_restore)
                    tmem_wait_st()
                    tri_apply(cf2, y4)
                    un = cute.make_rmem_tensor(4, Float32)
                    for r in cutlass.range_constexpr(4):
                        uu = y4[r] * isc
                        uh = f2h(uu)
                        stg16(head_b + Int64(17408 + 512 * r) + Int64(2) * Int64(t), uh)
                        un[r] = h2f(uh) * (-sc)
                    a, b = sw32_row(bo, t)
                    sts128(a, un[0].bitcast(Uint32), un[1].bitcast(Uint32), un[2].bitcast(Uint32), un[3].bitcast(Uint32))
                tc_before()
                named_bar(bar, 128)
                if leader:
                    fence_async_smem()
                    tc_after()
                    with cute.arch.elect_one():
                        mma_list(tmS, mbar_g, [ao >> Uint32(4), bo >> Uint32(4)], [("tf32", 0, "0xC000401000010000", 1, "0xC000401000010000", "0x08200910", True)])
                mbar_wait(mbar_g, mph)
                mph = mph ^ Uint32(1)
                tc_after()
                # ---------------------------------------------------------- P6: column sums of |E|
                part = cute.make_rmem_tensor(32, Float32)
                lane_off2 = (Uint32(32) * gw.to(Uint32) + Uint32(16)) << Uint32(16)
                vab = tmem_wait_ld(tmem_ld16x256x4_nw(tmS + lane_off) + tmem_ld16x256x4_nw(tmS + lane_off2))
                for blk in cutlass.range_constexpr(4):
                    va = vab[0:16]
                    vb = vab[16:32]
                    if cutlass.const_expr(blk < 3):
                        nxt = tmem_ld16x256x4_nw(tmS + lane_off + Uint32(32 * (blk + 1))) + tmem_ld16x256x4_nw(tmS + lane_off2 + Uint32(32 * (blk + 1)))
                    for m in cutlass.range_constexpr(4):
                        a0, a1 = fadd2(cute.math.abs(va[m * 4]), cute.math.abs(va[m * 4 + 1]),
                                       cute.math.abs(va[m * 4 + 2]), cute.math.abs(va[m * 4 + 3]))
                        b0, b1 = fadd2(cute.math.abs(vb[m * 4]), cute.math.abs(vb[m * 4 + 1]),
                                       cute.math.abs(vb[m * 4 + 2]), cute.math.abs(vb[m * 4 + 3]))
                        part[blk * 8 + m * 2], part[blk * 8 + m * 2 + 1] = fadd2(a0, a1, b0, b1)
                    if cutlass.const_expr(blk < 3):
                        vab = tmem_wait_ld(nxt)
                # thread (g8 = lane // 4, q = lane % 4) holds columns 8 m' + 2 q + c, m' = blk * 4 + m (index blk*8+m*2+c)
                # transposed reduction over g8 (xor 16, 8, 4) on fp16x2 pairs of adjacent columns: keep halves of the m'
                # range. Every column sum is < 128 max|S| < 2^14 (Cauchy-Schwarz on the projected column), so fp16 cannot
                # overflow; its rounding only perturbs the column scale, which stays consistent with the stored codes.
                ph = cute.make_rmem_tensor(16, Uint32)
                for j in cutlass.range_constexpr(16):
                    ph[j] = f2h2(part[2 * j], part[2 * j + 1])
                cnt = 16
                for lvl in cutlass.range_constexpr(3):
                    msk = 16 >> lvl
                    half = cnt // 2
                    upper = (lane & msk) != 0
                    for i in cutlass.range_constexpr(half):
                        send = ph[half + i]
                        keep = ph[i]
                        if upper:
                            send = ph[i]
                            keep = ph[half + i]
                        ph[i] = hadd2(keep, shfl_bfly(send.bitcast(Float32), msk).bitcast(Uint32))
                    cnt = half
                part[0], part[1] = h2f2(ph[0])
                part[2], part[3] = h2f2(ph[1])
                # now part[0..3] = sums for m'-group selected by lane bits (4,3,2) and columns 2q + c
                # m' index: bit4 (lane&16) selected the upper half of 16 m', bit3 -> next, bit2 -> next
                sel = ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2
                for i in cutlass.range_constexpr(4):
                    mp = sel + i // 2
                    col = mp * 8 + (lane % 4) * 2 + (i % 2)
                    sts32(cpg + Uint32(512) * gw.to(Uint32) + Uint32(4) * col.to(Uint32), part[i].bitcast(Uint32))
                named_bar(bar, 128)
                csum = Float32(0)
                for w4 in cutlass.range_constexpr(4):
                    csum = csum + lds32f(cpg + Uint32(512 * w4) + Uint32(4) * t.to(Uint32))
                # sk' = max(sqrt(mean |E|), eps) and its reciprocal from one rsqrt (sqrt(x) = x rsqrt(x); x = 0 -> eps)
                xm = csum * Float32(1.0 / 128.0)
                rq = rsqrt_approx(xm)
                skp = cute.arch.fmax(xm * rq, Float32(EPS) * sc_h)
                stg32(head_b + Int64(16384) + Int64(4) * Int64(t), (skp * isc_h).bitcast(Uint32))
                sts32(rsg + Uint32(4) * t.to(Uint32), cute.arch.fmin(rq, Float32(1.0 / EPS) * isc_h).bitcast(Uint32))
                named_bar(bar, 128)
                # ---------------------------------------------------------- P7: t = E / sk', row max, sv
                mxa = Float32(0)
                mxb = Float32(0)
                mxc = Float32(0)
                mxd = Float32(0)
                rcur = []
                for q4 in cutlass.range_constexpr(4):
                    rcur += lds128f(rsg + Uint32(16 * q4))
                v = tmem_wait_ld(tmem_ld16_nw(tmS + lane_off))
                for blk in cutlass.range_constexpr(8):
                    if cutlass.const_expr(blk < 7):
                        nxt = tmem_ld16_nw(tmS + lane_off + Uint32(16 * (blk + 1)))
                        rnxt = []
                        for q4 in cutlass.range_constexpr(4):
                            rnxt += lds128f(rsg + Uint32(4 * ((blk + 1) * 16 + q4 * 4)))
                    tv = []
                    for q4 in cutlass.range_constexpr(4):
                        t0, t1 = fmul2(v[q4 * 4], v[q4 * 4 + 1], rcur[q4 * 4], rcur[q4 * 4 + 1])
                        t2, t3 = fmul2(v[q4 * 4 + 2], v[q4 * 4 + 3], rcur[q4 * 4 + 2], rcur[q4 * 4 + 3])
                        if cutlass.const_expr(q4 % 2 == 0):
                            mxa = maxabs3(mxa, t0, t1)
                            mxb = maxabs3(mxb, t2, t3)
                        else:
                            mxc = maxabs3(mxc, t0, t1)
                            mxd = maxabs3(mxd, t2, t3)
                        tv += [t0, t1, t2, t3]
                    tmem_st16_nw(tmS + lane_off + Uint32(16 * blk), tv)
                    if cutlass.const_expr(blk < 7):
                        v = tmem_wait_ld(nxt)
                        rcur = rnxt
                tmem_wait_st()
                mx = cute.arch.fmax(cute.arch.fmax(mxa, mxb), cute.arch.fmax(mxc, mxd))
                svp = cute.arch.fmax(mx, Float32(EPS) * sc_h)
                stg32(head_b + Int64(16896) + Int64(4) * Int64(t), (svp * isc_h).bitcast(Uint32))
                # Hardware reciprocal in the normal range; retain accurate rounding at extreme row scales.
                rinv = rcp_approx(svp)
                if (svp <= Float32(1e-20)) | (svp >= Float32(1e20)):
                    rinv = rcp_rn(svp)
                inv = Float32(127.0) * rinv
                # ---------------------------------------------------------- P8: codes -> staging (SW128 rows) -> TMA store
                ctile = au + Uint32(4096) * gw.to(Uint32)
                if gw >= Int32(2):
                    ctile = ao + Uint32(4096) * (gw - Int32(2)).to(Uint32)
                cwx = (ctile + Uint32(128) * Uint32(lane)) | (tx7 << Uint32(4))
                v = tmem_wait_ld(tmem_ld16_nw(tmS + lane_off))
                for blk in cutlass.range_constexpr(8):
                    if cutlass.const_expr(blk < 7):
                        nxt = tmem_ld16_nw(tmS + lane_off + Uint32(16 * (blk + 1)))
                    wd = cute.make_rmem_tensor(4, Uint32)
                    for q4 in cutlass.range_constexpr(4):
                        x0, x1 = ffma2(v[q4 * 4], v[q4 * 4 + 1], inv, inv, Float32(MAGIC_RND), Float32(MAGIC_RND))
                        x2, x3 = ffma2(v[q4 * 4 + 2], v[q4 * 4 + 3], inv, inv, Float32(MAGIC_RND), Float32(MAGIC_RND))
                        lo = prmt(x0.bitcast(Uint32), x1.bitcast(Uint32), Uint32(0x0040))
                        hi = prmt(x2.bitcast(Uint32), x3.bitcast(Uint32), Uint32(0x0040))
                        wd[q4] = prmt(lo, hi, Uint32(0x5410))
                    sts128(cwx ^ Uint32(16 * blk), wd[0], wd[1], wd[2], wd[3])
                    if cutlass.const_expr(blk < 7):
                        v = tmem_wait_ld(nxt)
                fence_async_smem()
                cute.arch.sync_warp()
                if lane == 0:
                    tma_store_4d(tmap_c32, Int32(0), Int32(32) * gw, hh, slot, ctile)
                    bulk_commit()
                # next program for this group
                fph = fph ^ Uint32(1)
                mbar_wait(full0 + Uint32(8) * s.to(Uint32), fph)
                pid = lds32(misc0 + Uint32(4) * s.to(Uint32)).to(Int32)
            if lane == 0:
                bulk_wait0()
    tc_before()
    cute.arch.sync_threads()
    if warp == 0:
        allocated = Boolean(True)
        if cutlass.const_expr(early_exit):
            allocated = lds32(misc0 + Uint32(4 * (NSTAGE + 1))) != Uint32(0)
        if allocated:
            tc_after()
            tmem_free(tmem_base, 512)
            tmem_relinquish()


@cute.jit
def launch(raw: cute.Pointer, kbp: cute.Pointer, ubp: cute.Pointer, wp: cute.Pointer, pp: cute.Pointer,
           hcnt: cute.Pointer, idx: cute.Pointer, q0p: cute.Pointer, n: Int32, ns: Int32, tmaps: cute.Pointer,
           grid: Int32, stream: cuda.CUstream, early_exit: cutlass.Constexpr):
    flush_kernel(raw, kbp, ubp, wp, pp, hcnt, idx, q0p, n, ns, tmaps, early_exit).launch(
        grid=(grid, 1, 1), block=(NTHREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------
_KERNELS = {}


def _ptr(t, dtype=Float32):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.generic, assumed_align=16)


def _tensor_map(dtype, addr, dims, strides, box):
    res, tmap = cuda.cuTensorMapEncodeTiled(
        dtype, len(dims), addr, [cuda.cuuint64_t(d) for d in dims], [cuda.cuuint64_t(s) for s in strides],
        [cuda.cuuint32_t(b) for b in box], [cuda.cuuint32_t(1)] * len(dims),
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert res == cuda.CUresult.CUDA_SUCCESS, res
    return b''.join(int(x).to_bytes(8, 'little') for x in tmap.opaque)


class Pool:
    """Caller-owned state pool (the serving engine's layout). Every (slot, head) owns a 64 KB region of raw;
    bytes [0,16384) int8 codes [V,K], [16384,16896) fp32 s_k, [16896,17408) fp32 s_v, [17408,19456) fp16
    Compensator Tokens [R, K+V] (u | q). Slot stride SS words with one spare head; raw starts at OFF words.
    Slot 0 is reserved (idx 0 = no sequence). kbuf/ubuf/w/p/hcnt are dense [NS, HV, ...] tensors."""

    def __init__(self, num_slots, device="cuda"):
        NS = self.NS = num_slots
        self.raw = torch.zeros(NS * SS + OFF, dtype=torch.float32, device=device)
        st = self.raw.untyped_storage()

        def view(dtype, shape, strides, off):
            return torch.empty(0, dtype=dtype, device=device).set_(st).as_strided(shape, strides, off)
        self.codes = view(torch.int8, (NS, HV, V, K), (4 * SS, 4 * V * K, K, 1), 4 * OFF)
        self.codes32 = view(torch.int32, (NS, HV, V, K // 4), (SS, V * K, K // 4, 1), OFF)
        self.codes16 = view(torch.int16, (NS, HV, V, K // 2), (2 * SS, 2 * V * K, K // 2, 1), 2 * OFF)
        self.s_k = view(torch.float32, (NS, HV, K), (SS, V * K, 1), OFF + V * K // 4)
        self.s_v = view(torch.float32, (NS, HV, V), (SS, V * K, 1), OFF + V * K // 4 + K)
        self.s_kv = view(torch.float32, (NS, HV, K + V), (SS, V * K, 1), OFF + V * K // 4)
        self.uq = view(torch.float16, (NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), 2 * (OFF + V * K // 4 + K + V))
        self.kbuf = torch.zeros(NS, HV, L, K, dtype=torch.bfloat16, device=device)
        self.ubuf = torch.zeros(NS, HV, L, V, dtype=torch.bfloat16, device=device)
        self.w = torch.zeros(NS, HV, L, dtype=torch.float32, device=device)
        self.p = torch.ones(NS, HV, 1, dtype=torch.float32, device=device)
        self.hcnt = torch.zeros(NS, HV, dtype=torch.int32, device=device)
        dt = cuda.CUtensorMapDataType
        m_codes = _tensor_map(dt.CU_TENSOR_MAP_DATA_TYPE_UINT8, self.raw.data_ptr() + 4 * OFF,
                              [128, 128, HV + 1, NS], [128, 65536, 4 * SS], [128, 128, 1, 1])
        m_kb = _tensor_map(dt.CU_TENSOR_MAP_DATA_TYPE_UINT16, self.kbuf.data_ptr(),
                           [64, 16, 2, NS * HV], [256, 128, 4096], [64, 16, 2, 1])
        m_ub = _tensor_map(dt.CU_TENSOR_MAP_DATA_TYPE_UINT16, self.ubuf.data_ptr(),
                           [64, 16, 2, NS * HV], [256, 128, 4096], [64, 16, 2, 1])
        m_c32 = _tensor_map(dt.CU_TENSOR_MAP_DATA_TYPE_UINT8, self.raw.data_ptr() + 4 * OFF,
                            [128, 128, HV + 1, NS], [128, 65536, 4 * SS], [128, 32, 1, 1])
        self.tmaps = torch.frombuffer(bytearray(m_codes + m_kb + m_c32 + m_ub), dtype=torch.uint8).to(device)
        dev = self.raw.device
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        key = dev.index
        with torch.cuda.device(dev):
            if key not in _KERNELS:
                stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                args = (_ptr(self.raw), _ptr(self.kbuf, BFloat16), _ptr(self.ubuf, BFloat16), _ptr(self.w), _ptr(self.p),
                        _ptr(self.hcnt, Int32), _ptr(self.hcnt, Int32), _ptr(self.w), Int32(1), Int32(1), _ptr(self.tmaps, Int8),
                        Int32(1), stream)
                _KERNELS[key] = tuple(cute.compile(launch, *args, flag, options="--enable-tvm-ffi") for flag in (True, False))
        self.kernel, self.small_kernel = _KERNELS[key]


def flush_inplace(pool, idx, q0):
    """Flush every (idx[n], head) whose fill counter equals L (idx 0 = skip): new checkpoint in place, w = 0, p = 1."""
    n = idx.numel()
    if n == 0:
        return
    if idx.dtype != torch.int32 or not idx.is_contiguous() or idx.data_ptr() % 16:
        idx = idx.to(torch.int32).contiguous().clone()
    if q0.dtype != torch.float32 or not q0.is_contiguous() or q0.data_ptr() % 16:
        q0 = q0.to(torch.float32).contiguous().clone()
    grid = min(pool.sms, n * HV)
    with torch.cuda.device(pool.raw.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        kernel = pool.small_kernel if n * HV < 16 * pool.sms else pool.kernel
        kernel(_ptr(pool.raw), _ptr(pool.kbuf, BFloat16), _ptr(pool.ubuf, BFloat16), _ptr(pool.w), _ptr(pool.p),
                    _ptr(pool.hcnt, Int32), _ptr(idx, Int32), _ptr(q0), Int32(n), Int32(pool.NS), _ptr(pool.tmaps, Int8), Int32(grid), stream)


_POOLS = {}


def run(codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0):
    """Functional form: pack into a pool, flush, return the new checkpoint as fresh tensors."""
    B = codes.shape[0]
    key = (B, codes.device)
    pool = _POOLS.get(key)
    if pool is None:
        pool = _POOLS[key] = Pool(B + 8, codes.device)
    sl = slice(1, B + 1)
    pool.codes[sl] = codes
    pool.s_k[sl] = s_k
    pool.s_v[sl] = s_v
    pool.uq[sl, :, :, :K] = u
    pool.uq[sl, :, :, K:] = q
    pool.kbuf[sl] = kbuf
    pool.ubuf[sl] = ubuf
    pool.w[sl] = w
    pool.p[sl, :, 0] = p
    pool.hcnt.zero_()
    pool.hcnt[sl] = L
    flush_inplace(pool, torch.arange(1, B + 1, dtype=torch.int32, device=codes.device), q0.contiguous())
    return (pool.codes[sl].clone(), pool.s_k[sl].clone(), pool.s_v[sl].clone(),
            pool.uq[sl, :, :, :K].clone(), pool.uq[sl, :, :, K:].clone())
