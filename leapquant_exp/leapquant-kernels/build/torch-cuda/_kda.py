# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL window flush for the caller-owned LeapQuant KDA pool (v2 layout).

One persistent CTA of four warps per program slot, four CTAs per SM.  The fp32 state lives in tensor memory with
its key columns permuted (TMEM column n holds key pi(n)) so that the row-wise passes, the column reduction and the
code stores all use cheap register layouts.  Inputs arrive by bulk/TMA copies, the new codes leave by a TMA store.
The start basis of the subspace iteration enters its first skinny product as fp16 high + low parts.
Pool setup compiles three variants that differ only in L2 eviction hints and, for calls of up to 64 sequences, in the
startup (the first window is sent before the setup barriers) and in where the window resets are issued.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float16, BFloat16, Int8, Int32, Int64, Uint64, TFloat32
from cutlass.cute.runtime import make_ptr
from cutlass.utils import SmemAllocator
from cutlass._mlir.dialects import llvm, arith
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass.cute.nvgpu import tcgen05 as tc
from cutlass.cute.nvgpu import cpasync
from cuda.bindings import driver as cuda

HV, K, V, L, R = 32, 128, 128, 16, 4
SS, OFF = 33 * V * K, 12288
CTAS_PER_SM = 4
MAXDUE = 128
TAU = 2.0 ** -9          # one CholeskyQR pass suffices when every normalized pivot exceeds TAU

# shared-memory map (bytes from a 1024-aligned base)
SF = 0              # 32 KB: fp16 state copy | old-factor operands | codes in (upper half) | |E| bf16 | codes out
KB = 32768          # raw kbuf -> keys hi operand
GB = 36864          # raw gbuf -> keys lo operand
UB = 40960          # raw ubuf -> window A operand
META = 45056        # s_k (512) | s_v (512) | uq (2048)
PCM = 48128         # pcum (512)
WGT = 48640         # w (64)
SKB = 48768         # skinny B operand, 128 x 8 halves
QVB = 50816         # Q rows (fp32 128 x 4) for the QR
ONES = 52864        # 1 KB of fp16 ones: source of the bulk gbuf window resets
SKP = 53888         # per-TMEM-column scale (skpc' then 1/sk')
BND = 54400         # bound partials 4 warps x 8
DUE = 54528         # due list
CNT = DUE + 4 * MAXDUE
MBAR = CNT + 16     # 3 mbarriers: mma, window, codes
TMH = MBAR + 24
SEEDB = (TMH + 8 + 127) // 128 * 128   # conditioned start basis as the first skinny B operand (128 x 8 halves)
SMEM_BYTES = SEEDB + 2048
WIN_BYTES = 4096 * 3 + 3072 + 512 + 64


def encode_map(ptr, dims, strides, box, dtype, swizzle, device):
    status, mapping = cuda.cuTensorMapEncodeTiled(dtype, len(dims), ptr,
        tuple(cuda.cuuint64_t(x) for x in dims), tuple(cuda.cuuint64_t(x) for x in strides),
        tuple(cuda.cuuint32_t(x) for x in box), tuple(cuda.cuuint32_t(1) for _ in dims),
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert status == cuda.CUresult.CUDA_SUCCESS, status
    return torch.tensor([int(x) for x in mapping.opaque], dtype=torch.uint64, device=device)


def _asm(text, cons, args=(), loc=None, ip=None, ret=None):
    return llvm.inline_asm(ret, list(args), text, cons, has_side_effects=True, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tma_acquire(desc, *, loc=None, ip=None):
    _asm('fence.proxy.tensormap::generic.acquire.gpu [$0],128;', 'l,~{memory}', [Int64(desc).ir_value(loc=loc, ip=ip)], loc, ip)


@dsl_user_op
def tma_load_codes(dst, desc, head, slot, mbar, *, loc=None, ip=None):
    args = [Int32(dst).ir_value(loc=loc, ip=ip), Int64(desc).ir_value(loc=loc, ip=ip), Int32(head).ir_value(loc=loc, ip=ip),
            Int32(slot).ir_value(loc=loc, ip=ip), Int32(mbar).ir_value(loc=loc, ip=ip)]
    _asm('cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {0, 0, $2, $3}], [$4];',
         'r,l,r,r,r,~{memory}', args, loc, ip)


@dsl_user_op
def tma_store_codes(src, desc, head, slot, *, loc=None, ip=None):
    args = [Int64(desc).ir_value(loc=loc, ip=ip), Int32(head).ir_value(loc=loc, ip=ip),
            Int32(slot).ir_value(loc=loc, ip=ip), Int32(src).ir_value(loc=loc, ip=ip)]
    _asm('cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [$0, {0, 0, $1, $2}], [$3];', 'l,r,r,r,~{memory}', args, loc, ip)


@dsl_user_op
def bulk_load(dst, src, nbytes: int, mbar, fraction: float, *, loc=None, ip=None):
    args = [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip), Int32(mbar).ir_value(loc=loc, ip=ip)]
    if fraction is None:
        _asm(f'cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];', 'r,l,r,~{memory}', args, loc, ip)
    else:
        args.append(l2_policy(fraction, loc=loc, ip=ip).ir_value(loc=loc, ip=ip))
        _asm(f'cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], {nbytes}, [$2], $3;', 'r,l,r,l,~{memory}', args, loc, ip)


@dsl_user_op
def mbar_expect(mbar, nbytes: int, *, loc=None, ip=None):
    _asm(f'mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};', 'r,~{memory}', [Int32(mbar).ir_value(loc=loc, ip=ip)], loc, ip)


@dsl_user_op
def mbar_wait(mbar, phase, *, loc=None, ip=None):
    """mbarrier wait followed by a compiler memory fence (plain loads must not be hoisted above it)."""
    cute.arch.mbarrier_wait(mbar, phase, loc=loc, ip=ip)
    _asm('', '~{memory}', [], loc, ip)


@dsl_user_op
def mbar_spin(mbar, phase, *, loc=None, ip=None):
    """mbarrier phase wait as a test_wait spin, then a compiler memory fence.  Used for the tensor-core products,
    which complete within a few hundred cycles: the suspending try_wait of cute.arch.mbarrier_wait wakes up late."""
    args = [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)]
    _asm('{ .reg .pred p; WAIT_${:uid}: mbarrier.test_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT_${:uid}; }', 'r,r,~{memory}', args, loc, ip)


@dsl_user_op
def prefetch_l2(src, nbytes: int, *, loc=None, ip=None):
    _asm(f'cp.async.bulk.prefetch.L2.global [$0], {nbytes};', 'l,~{memory}', [Int64(src).ir_value(loc=loc, ip=ip)], loc, ip)


@dsl_user_op
def bulk_store(dst, src, nbytes: int, fraction: float, *, loc=None, ip=None):
    args = [Int64(dst).ir_value(loc=loc, ip=ip), Int32(src).ir_value(loc=loc, ip=ip)]
    if fraction is None:
        _asm(f'cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], {nbytes};', 'l,r,~{memory}', args, loc, ip)
    else:
        args.append(l2_policy(fraction, loc=loc, ip=ip).ir_value(loc=loc, ip=ip))
        _asm(f'cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [$0], [$1], {nbytes}, $2;', 'l,r,l,~{memory}', args, loc, ip)


@dsl_user_op
def bulk_commit(*, loc=None, ip=None):
    _asm('cp.async.bulk.commit_group;', '~{memory}', [], loc, ip)


@dsl_user_op
def bulk_wait(*, loc=None, ip=None):
    _asm('cp.async.bulk.wait_group 0;', '~{memory}', [], loc, ip)


@dsl_user_op
def bulk_wait_read(*, loc=None, ip=None):
    _asm('cp.async.bulk.wait_group.read 0;', '~{memory}', [], loc, ip)


@dsl_user_op
def fence_async_smem(*, loc=None, ip=None):
    _asm('fence.proxy.async.shared::cta;', '~{memory}', [], loc, ip)


@dsl_user_op
def tc_before(*, loc=None, ip=None):
    _asm('tcgen05.fence::before_thread_sync;', '~{memory}', [], loc, ip)


@dsl_user_op
def tc_after(*, loc=None, ip=None):
    _asm('tcgen05.fence::after_thread_sync;', '~{memory}', [], loc, ip)


@dsl_user_op
def tc_wait_st(*, loc=None, ip=None):
    _asm('tcgen05.wait::st.sync.aligned;', '~{memory}', [], loc, ip)


@dsl_user_op
def tmem_load(addr, shape: str, count: int, *, loc=None, ip=None):
    ty = ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * count) + ')>')
    outs = ','.join(f'${i}' for i in range(count))
    out = _asm(f'tcgen05.ld.sync.aligned.{shape}.b32 {{{outs}}}, [${count}]; tcgen05.wait::ld.sync.aligned;',
               ','.join(['=f'] * count + ['r']), [Int32(addr).ir_value(loc=loc, ip=ip)], loc, ip, ty)
    return [Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(count)]


@dsl_user_op
def tmem_load_nowait(addr, shape: str, count: int, *, loc=None, ip=None):
    """tcgen05.ld without the wait (tc_wait_ld must precede any use of the values)."""
    ty = ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * count) + ')>')
    outs = ','.join(f'${i}' for i in range(count))
    out = _asm(f'tcgen05.ld.sync.aligned.{shape}.b32 {{{outs}}}, [${count}];',
               ','.join(['=f'] * count + ['r']), [Int32(addr).ir_value(loc=loc, ip=ip)], loc, ip, ty)
    return [Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(count)]


@dsl_user_op
def tc_wait_ld(*, loc=None, ip=None):
    _asm('tcgen05.wait::ld.sync.aligned;', '~{memory}', [], loc, ip)


@dsl_user_op
def tmem_store(addr, shape: str, values, *, loc=None, ip=None):
    count = len(values)
    args = [Int32(addr).ir_value(loc=loc, ip=ip)] + [Float32(x).ir_value(loc=loc, ip=ip) for x in values]
    regs = ','.join(f'${i + 1}' for i in range(count))
    _asm(f'tcgen05.st.sync.aligned.{shape}.b32 [$0], {{{regs}}};', ','.join(['r'] + ['f'] * count + ['~{memory}']), args, loc, ip)


@dsl_user_op
def sts128(addr, a, b, c, d, *, loc=None, ip=None):
    args = [Int32(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)]
    _asm('st.shared.v4.b32 [$0], {$1,$2,$3,$4};', 'r,r,r,r,r,~{memory}', args, loc, ip)


def _smem_vec(dtype, addr, n, align):
    t = cute.make_tensor(cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align), cute.make_layout(n))
    r = cute.make_rmem_tensor(n, dtype)
    cute.autovec_copy(t, r)
    return [r[i] for i in range(n)]


def lds128(addr):
    """Plain (compiler-schedulable) 16-byte shared load as four Int32."""
    return _smem_vec(Int32, addr, 4, 16)


def lds128f(addr):
    return _smem_vec(Float32, addr, 4, 16)


def lds64f(addr):
    return _smem_vec(Float32, addr, 2, 8)


def lds_u16(addr):
    return cute.make_tensor(cute.make_ptr(cutlass.Uint16, addr, cute.AddressSpace.smem, assumed_align=2), cute.make_layout(1))[0].to(Int32) & Int32(0xFFFF)


@dsl_user_op
def stsm4(addr, a, b, c, d, *, loc=None, ip=None):
    args = [Int32(addr).ir_value(loc=loc, ip=ip)] + [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)]
    _asm('stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0],{$1,$2,$3,$4};', 'r,r,r,r,r,~{memory}', args, loc, ip)


@dsl_user_op
def bits_float(x, *, loc=None, ip=None):
    return Float32(arith.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def float_bits(x, *, loc=None, ip=None):
    return Int32(arith.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _pure(text, cons, args, ret, loc, ip):
    return llvm.inline_asm(ret, list(args), text, cons, has_side_effects=False, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def l2_policy(fraction: float, *, loc=None, ip=None):
    """None keeps the default policy; a fraction gives early eviction priority."""
    if fraction is None:
        return None
    return Int64(_pure(f'createpolicy.fractional.L2::evict_first.b64 $0, {fraction};', '=l', [], T.i64(), loc, ip))


@dsl_user_op
def pack_half(a, b, *, loc=None, ip=None):
    """fp16x2 word: a in the low half."""
    return Int32(_pure('cvt.rn.f16x2.f32 $0,$2,$1;', '=r,f,f', [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip))


@dsl_user_op
def pack_bf16(a, b, *, loc=None, ip=None):
    """bf16x2 word: a in the low half."""
    return Int32(_pure('cvt.rn.bf16x2.f32 $0,$2,$1;', '=r,f,f', [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip))


@dsl_user_op
def prmt(a, b, sel: int, *, loc=None, ip=None):
    return Int32(_pure(f'prmt.b32 $0,$1,$2,{sel:#x};', '=r,r,r', [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip))


@dsl_user_op
def sext_byte(w, b: int, *, loc=None, ip=None):
    """Byte b of w sign-extended to 32 bits (one PRMT)."""
    sel = 0x8880 | b | ((b | 8) << 4) | ((b | 8) << 8) | ((b | 8) << 12)
    return Int32(_pure(f'prmt.b32 $0,$1,0,{sel:#x};', '=r,r', [Int32(w).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip))


@dsl_user_op
def magic_byte(wx, b: int, *, loc=None, ip=None):
    """Float 32768 + byte b of wx (byte placed in mantissa bits 8..15 under exponent 2^15), one PRMT."""
    sel = 0x7404 | (b << 4)
    return bits_float(Int32(_pure(f'prmt.b32 $0,$1,0x47000000,{sel:#x};', '=r,r', [Int32(wx).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip)))


@dsl_user_op
def max_u16x2(a, b, *, loc=None, ip=None):
    return Int32(_pure('max.u16x2 $0,$1,$2;', '=r,r,r', [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip))


@dsl_user_op
def f16x2_f32(w, *, loc=None, ip=None):
    """The two fp16 halves of w as fp32 (low half first)."""
    ty = ir.Type.parse('!llvm.struct<(f32,f32)>')
    out = _pure('{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.f32.f16 $0, l; cvt.f32.f16 $1, h; }', '=f,=f,r',
                [Int32(w).ir_value(loc=loc, ip=ip)], ty, loc, ip)
    return Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)), Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip))


@dsl_user_op
def redux_max(x, *, loc=None, ip=None):
    """Warp maximum of a nonnegative float (unsigned order of its bits)."""
    raw = arith.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    out = _asm('redux.sync.max.u32 $0,$1,0xffffffff;', '=r,r', [raw], loc, ip, T.i32())
    return Float32(arith.bitcast(T.f32(), out, loc=loc, ip=ip))


@dsl_user_op
def tf32_bits(x, *, loc=None, ip=None):
    return Int32(_pure('cvt.rn.tf32.f32 $0,$1;', '=r,f',
                      [Float32(x).ir_value(loc=loc, ip=ip)], T.i32(), loc, ip))


@dsl_user_op
def quant_pack(a, b, c, d, scale, *, loc=None, ip=None):
    """Round a..d * scale to nearest (even) integers and pack their low bytes, a in byte 0."""
    bias = Float32(12582912)
    q0, q1 = cute.arch.fma_packed_f32x2((Float32(a), Float32(b)), (Float32(scale), Float32(scale)), (bias, bias), ftz=False, loc=loc, ip=ip)
    q2, q3 = cute.arch.fma_packed_f32x2((Float32(c), Float32(d)), (Float32(scale), Float32(scale)), (bias, bias), ftz=False, loc=loc, ip=ip)
    args = [x.ir_value(loc=loc, ip=ip) for x in (q0, q1, q2, q3)]
    return Int32(_pure('{ .reg .b32 p0,p1; prmt.b32 p0,$1,$2,0x0040; prmt.b32 p1,$3,$4,0x0040; prmt.b32 $0,p0,p1,0x5410; }',
                       '=r,f,f,f,f', args, T.i32(), loc, ip))


def pi_inv(t):
    """TMEM column of key t: k = 16m + 4c + 2h + i -> n = 16m + 8h + 2c + i."""
    return (t & 0x71) | ((t & 2) << 2) | ((t >> 1) & 6)


def pi_of(n):
    """Key held by TMEM column n."""
    return (n & 0x71) | ((n >> 2) & 2) | ((n << 1) & 12)


def sw32(base, row, chunk):
    """Byte address of 16-byte chunk (0/1) of a 32-byte row in a 128B-aligned SW32 tile."""
    return base + row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4)


def sw128(base, row, chunk):
    """Byte address of 16-byte chunk (0..7) of a 128-byte row in a 1024B-aligned SW128 tile."""
    return base + row * 128 + ((chunk ^ (row & 7)) << 4)


@cute.jit
def seed_load(a: cute.Pointer, tid: Int32):
    """The start matrix rows tid + 32i into warp 0's registers (issued at kernel entry)."""
    z = cute.make_rmem_tensor((4, 4), Float32)
    for i in cutlass.range_constexpr(4):
        for r in cutlass.range_constexpr(4):
            z[r, i] = Float32(0)
    if cute.arch.make_warp_uniform(tid // 32) == 0:
        for i in cutlass.range_constexpr(4):
            ap = cute.make_ptr(Float32, a.toint() + (tid + 32 * i) * 16, cute.AddressSpace.gmem, assumed_align=16)
            row = cute.make_rmem_tensor(4, Float32)
            cute.autovec_copy(cute.make_tensor(ap, cute.make_layout(4)), row)
            for r in cutlass.range_constexpr(4):
                z[r, i] = row[r]
    return z


@cute.jit
def seed_scale(z: cute.Tensor, qv_addr: Int32, tid: Int32):
    """Start matrix rows to qv (fp32, warp 0), each column scaled by an exact power of two (orthogonalize follows)."""
    if cute.arch.make_warp_uniform(tid // 32) == 0:
        for r in cutlass.range_constexpr(4):
            peak = Float32(0)
            for i in cutlass.range_constexpr(4):
                peak = cute.arch.fmax(peak, cute.math.abs(z[r, i]))
            peak = redux_max(peak)
            eb = cutlass.min(cutlass.max(Int32(254) - ((float_bits(peak) >> 23) & 255), Int32(1)), Int32(252))
            scale = bits_float(eb << 23)
            for i in cutlass.range_constexpr(4):
                z[r, i] *= scale
        for i in cutlass.range_constexpr(4):
            row = cute.make_rmem_tensor(4, Float32)
            for r in cutlass.range_constexpr(4):
                row[r] = z[r, i]
            cute.autovec_copy(row, cute.make_tensor(cute.make_ptr(Float32, qv_addr + (tid + 32 * i) * 16, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
        cute.arch.sync_warp()


@cute.jit
def chol_pass(z: cute.Tensor, qv_addr: Int32, tid: Int32, tau: Float32):
    """One CholeskyQR pass with relative pivot floors on warp 0's rows z (4 per lane, in place).  Returns True when
    every pivot keeps more than tau of its column's squared norm (the pass is then accurate to about 1e-4)."""
    g = cute.make_rmem_tensor(10, Float32)
    for r in cutlass.range_constexpr(4):
        for c in cutlass.range_constexpr(r + 1):
            dot = Float32(0)
            for i in cutlass.range_constexpr(4):
                dot += z[r, i] * z[c, i]
            g[r * (r + 1) // 2 + c] = dot
    # warp sums of the 10 Gram partials through shared memory (rows of 36 floats at qv: the Q rows are
    # already in registers): lane e < 10 adds up entry e, then every lane reads the 10 totals
    cute.arch.sync_warp()
    for i in cutlass.range_constexpr(10):
        scr = cute.make_ptr(Float32, qv_addr + (i * 36 + tid) * 4, cute.AddressSpace.smem, assumed_align=4)
        cute.make_tensor(scr, cute.make_layout(1))[0] = g[i]
    cute.arch.sync_warp()
    row = qv_addr + cutlass.min(tid, Int32(9)) * 144
    part = lds128f(row)
    for j in cutlass.range_constexpr(1, 8):
        nxt = lds128f(row + 16 * j)
        for k in cutlass.range_constexpr(4):
            part[k] = part[k] + nxt[k]
    tot = (part[0] + part[1]) + (part[2] + part[3])
    if tid < 10:
        cute.make_tensor(cute.make_ptr(Float32, qv_addr + (360 + tid) * 4, cute.AddressSpace.smem, assumed_align=4), cute.make_layout(1))[0] = tot
    cute.arch.sync_warp()
    gt = lds128f(qv_addr + 1440) + lds128f(qv_addr + 1456) + lds64f(qv_addr + 1472)
    for i in cutlass.range_constexpr(10):
        g[i] = gt[i]
    d0 = cute.math.rsqrt(cute.arch.fmax(g[0], Float32(1e-30)), fastmath=True)
    l10 = g[1] * d0
    p1 = g[2] - l10 * l10
    d1 = cute.math.rsqrt(cute.arch.fmax(cute.arch.fmax(p1, Float32(1e-6) * g[2]), Float32(1e-30)), fastmath=True)
    l20 = g[3] * d0
    l21 = (g[4] - l20 * l10) * d1
    p2 = g[5] - l20 * l20 - l21 * l21
    d2 = cute.math.rsqrt(cute.arch.fmax(cute.arch.fmax(p2, Float32(1e-6) * g[5]), Float32(1e-30)), fastmath=True)
    l30 = g[6] * d0
    l31 = (g[7] - l30 * l10) * d1
    l32 = (g[8] - l30 * l20 - l31 * l21) * d2
    p3 = g[9] - l30 * l30 - l31 * l31 - l32 * l32
    d3 = cute.math.rsqrt(cute.arch.fmax(cute.arch.fmax(p3, Float32(1e-6) * g[9]), Float32(1e-30)), fastmath=True)
    for i in cutlass.range_constexpr(4):
        z[0, i] *= d0
        z[1, i] = (z[1, i] - z[0, i] * l10) * d1
        z[2, i] = (z[2, i] - z[0, i] * l20 - z[1, i] * l21) * d2
        z[3, i] = (z[3, i] - z[0, i] * l30 - z[1, i] * l31 - z[2, i] * l32) * d3
    margin = cute.arch.fmin(cute.arch.fmin(p1 - tau * g[2], p2 - tau * g[5]), p3 - tau * g[9])
    return (margin > Float32(0)) & (g[0] > Float32(1e-30))


@cute.jit
def orthogonalize(qv_addr: Int32, sk_addr: Int32, tid: Int32, tau: Float32, lo: cutlass.Constexpr):
    """CholeskyQR on the 128 x 4 fp32 rows at qv (warp 0); fp16 Q rows (lo: plus their fp16 remainders in columns 4-7)
    to the skinny tile.  The second pass only runs
    when the first one met a pivot below tau of its column's squared norm (ill-conditioned or rank-deficient
    bases): one pass leaves an orthogonality defect of about 2e-7 / min normalized pivot."""
    if cute.arch.make_warp_uniform(tid // 32) == 0:
        z = cute.make_rmem_tensor((4, 4), Float32)
        for i in cutlass.range_constexpr(4):
            vals = lds128f(qv_addr + (tid + 32 * i) * 16)
            for r in cutlass.range_constexpr(4):
                z[r, i] = vals[r]
        good = chol_pass(z, qv_addr, tid, tau)
        if not good:
            chol_pass(z, qv_addr, tid, tau)
        for i in cutlass.range_constexpr(4):
            sp = cute.make_ptr(Int32, sk_addr + (tid + 32 * i) * 16, cute.AddressSpace.smem, assumed_align=16)
            if cutlass.const_expr(lo):
                # start basis: columns 0-3 the fp16 high parts, 4-7 their remainders (the subspace iteration's
                # result is sensitive to its start in ill-conditioned heads)
                packed = cute.make_rmem_tensor(4, Int32)
                packed[0], packed[1] = pack_half(z[0, i], z[1, i]), pack_half(z[2, i], z[3, i])
                h0, h1 = f16x2_f32(packed[0]), f16x2_f32(packed[1])
                lo0 = cute.arch.sub_packed_f32x2((z[0, i], z[1, i]), h0, ftz=False)
                lo1 = cute.arch.sub_packed_f32x2((z[2, i], z[3, i]), h1, ftz=False)
                packed[2], packed[3] = pack_half(*lo0), pack_half(*lo1)
                cute.autovec_copy(packed, cute.make_tensor(sp, cute.make_layout(4)))
            else:
                # the new Q: fp16 rows in columns 0-3 (columns 4-7 of the skinny tile stay zero)
                packed = cute.make_rmem_tensor(2, Int32)
                packed[0], packed[1] = pack_half(z[0, i], z[1, i]), pack_half(z[2, i], z[3, i])
                cute.autovec_copy(packed, cute.make_tensor(sp, cute.make_layout(2)))


@cute.jit
def issue_copy_pair(base: Int32, off0: int, src0: Int64, n0: int, off1: int, src1: Int64, n1: int, mbar: Int32, fraction: cutlass.Constexpr):
    """Two bulk copies into shared memory, one expect + arrival on the window mbarrier."""
    mbar_expect(mbar, n0 + n1)
    bulk_load(base + off0, src0, n0, mbar, fraction)
    bulk_load(base + off1, src1, n1, mbar, fraction)


@cute.jit
def issue_window(base: Int32, kb: cute.Pointer, gb: cute.Pointer, wb: cute.Pointer, pb: cute.Pointer,
                 raw: cute.Pointer, ph: Int32, mbar: Int32, mb_win: cute.Pointer, warp: Int32, lane: Int32,
                 tma_atom_u: cute.CopyAtom, t_gu: cute.Tensor, t_su: cute.Tensor, fraction: cutlass.Constexpr):
    """Program ph's window by warps 1-3 (three arrivals; warp 0 issues the products, and a bulk-copy issue keeps a
    warp busy for ~150 cycles): two bulk copies on lane 0 of warps 1 / 2; warp 3 sends ubuf through the TMA atom
    (MN-major window operand as loaded) and META by a bulk copy."""
    pw = ph.to(Int64)
    if warp == 1:
        if lane == 0:
            issue_copy_pair(base, KB, kb.toint() + pw * 4096, 4096, WGT, wb.toint() + pw * 64, 64, mbar, fraction)
    if warp == 2:
        if lane == 0:
            issue_copy_pair(base, GB, gb.toint() + pw * 4096, 4096, PCM, pb.toint() + pw * 512, 512, mbar, fraction)
    if warp == 3:
        if lane == 0:
            rp = raw.toint() + (Int64(OFF) + (ph // 32).to(Int64) * SS + (ph % 32).to(Int64) * 16384) * 4
            mbar_expect(mbar, 4096 + 3072)
            bulk_load(base + META, rp + 16384, 3072, mbar, fraction)
        cute.copy(tma_atom_u, t_gu[(None, ph)], t_su, tma_bar_ptr=mb_win, cache_policy=l2_policy(fraction))


@cute.jit
def reset_window(base: Int32, gb: cute.Pointer, wb: cute.Pointer, pb: cute.Pointer, ph: Int32, tid: Int32, fraction: cutlass.Constexpr):
    """Program ph's window reset: pcum and w to zero, gbuf to ones by four bulk copies from the ones buffer (thread 0;
    the bulk group is waited with the code store's)."""
    logs = cute.make_tensor(pb + ph.to(Int64) * 128, cute.make_layout(128))
    logs[tid] = Float32(0)
    if tid < 16:
        weights = cute.make_tensor(wb + ph.to(Int64) * 16, cute.make_layout(16))
        weights[tid] = Float32(0)
    if tid == 0:
        for q in cutlass.range_constexpr(4):
            bulk_store(gb.toint() + ph.to(Int64) * 4096 + q * 1024, base + ONES, 1024, fraction)
        bulk_commit()


@cute.jit
def issue_codes(base: Int32, desc_addr: Int64, ph: Int32, mbar: Int32):
    mbar_expect(mbar, 16384)
    tma_load_codes(base + SF + 16384, desc_addr, ph % 32, ph // 32, mbar)


@cute.kernel
def flush_kernel(raw_addr: Int64, kb_addr: Int64, ub_addr: Int64, gb_addr: Int64,
                 wb_addr: Int64, pb_addr: Int64, hc_addr: Int64, desc_addr: Int64, start_addr: Int64,
                 idx_addr: Int64, dbg_addr: Int64, n: Int32, tma_atom_c: cute.CopyAtom, tma_c: cute.Tensor,
                 tma_atom_s: cute.CopyAtom, tma_s: cute.Tensor, tma_atom_u: cute.CopyAtom, tma_u: cute.Tensor,
                 blocks: cutlass.Constexpr, dbg: cutlass.Constexpr, fraction: cutlass.Constexpr = None):
    raw = cute.make_ptr(Float32, raw_addr, cute.AddressSpace.gmem, assumed_align=16)
    kb = cute.make_ptr(BFloat16, kb_addr, cute.AddressSpace.gmem, assumed_align=16)
    ub = cute.make_ptr(BFloat16, ub_addr, cute.AddressSpace.gmem, assumed_align=16)
    gb = cute.make_ptr(Float16, gb_addr, cute.AddressSpace.gmem, assumed_align=16)
    wb = cute.make_ptr(Float32, wb_addr, cute.AddressSpace.gmem, assumed_align=16)
    pb = cute.make_ptr(Float32, pb_addr, cute.AddressSpace.gmem, assumed_align=16)
    hc = cute.make_ptr(Int32, hc_addr, cute.AddressSpace.gmem, assumed_align=16)
    start = cute.make_ptr(Float32, start_addr, cute.AddressSpace.gmem, assumed_align=16)
    indices = cute.make_ptr(Int32, idx_addr, cute.AddressSpace.gmem, assumed_align=16)
    dbgp = cute.make_ptr(Float32, dbg_addr, cute.AddressSpace.gmem, assumed_align=16)
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane, warp = tid % 32, cute.arch.make_warp_uniform(tid // 32)
    smem = SmemAllocator()
    buf = smem.allocate_tensor(Int8, cute.make_layout(SMEM_BYTES), byte_alignment=1024)
    base = buf.iterator.toint().to(Int32)
    due = cute.make_tensor(cute.make_ptr(Int32, base + DUE, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(MAXDUE))
    counts = cute.make_tensor(cute.make_ptr(Int32, base + CNT, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4))
    bnd = cute.make_tensor(cute.make_ptr(Float32, base + BND, cute.AddressSpace.smem, assumed_align=16), cute.make_layout((4, 8), stride=(8, 1)))
    mb_mma = cute.make_ptr(Uint64, base + MBAR, cute.AddressSpace.smem, assumed_align=8)
    mb_win = cute.make_ptr(Uint64, base + MBAR + 8, cute.AddressSpace.smem, assumed_align=8)
    mb_codes = cute.make_ptr(Uint64, base + MBAR + 16, cute.AddressSpace.smem, assumed_align=8)
    tmh = cute.make_ptr(Int32, base + TMH, cute.AddressSpace.smem, assumed_align=4)
    skp = cute.make_tensor(cute.make_ptr(Float32, base + SKP, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(128))
    if warp == 0:                          # first: the next CTA on this SM launches once the permit is relinquished
        cute.arch.alloc_tmem(128, tmh)
        cute.arch.relinquish_tmem_alloc_permit()
    if tid == 32:                          # the shared start matrix into L2 ahead of the first windows' DRAM burst
        prefetch_l2(start_addr, 2048)
    if cutlass.const_expr(fraction is not None):
        # small calls: the start matrix rows (warp 0) load during the due selection, and warps 1-3 may send the first
        # window before the setup barriers (mb_win is initialized here for them)
        zseed_e = seed_load(start, tid)
        if tid == 32:
            cute.arch.mbarrier_init(mb_win, 3)
            cute.arch.mbarrier_init_fence()
        if warp != 0:
            cute.arch.barrier(barrier_id=1, number_of_threads=96)
        ub_layout_e = cute.tile_to_shape(tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.MN_SW128, BFloat16), (128, 16), order=(1, 0))
        ub_tile_e = cute.make_tensor(cute.make_ptr(BFloat16, base + UB, cute.AddressSpace.smem, assumed_align=1024, swizzle_=ub_layout_e.inner),
                                     ub_layout_e.outer)
        g_u_e = cute.local_tile(tma_u, (128, 16), (0, 0, None))
        t_su_e, t_gu_e = cpasync.tma_partition(tma_atom_u, 0, cute.make_layout(1), cute.group_modes(ub_tile_e, 0, 2), cute.group_modes(g_u_e, 0, 2))
    # GPU-side due-selection and start-matrix loads are in flight during setup.
    ix = cute.make_tensor(indices, cute.make_layout(n))
    hcnt = cute.make_tensor(hc, cute.make_layout(n * 32 + 1))
    # Weighted round-robin: the k-th program of CTA b is the rank of (2k + 1) / W[wave(b)] in a global virtual-time
    # order (ties: lower wave, then lower index), so every round stays an interleaved compact block of programs while
    # the CTAs launched first on each SM (wave = bid // SMs; they run faster under the scheduler's age priority) get
    # proportionally more programs.  Equal weights give p = bid + k * blocks.  The skew only applies from 6
    # programs per CTA on.
    nprog = n * 32
    p = bid + tid * blocks
    if nprog >= 6 * blocks and nprog < 100 * blocks:     # 128 candidates per CTA cover the weighted ranks below 100
        sms_c = blocks // 4
        wave_i = bid // sms_c
        t2 = 2 * tid + 1
        base_p = tid * sms_c + (bid - wave_i * sms_c)
        strong = nprog < 12 * blocks           # fewer programs per CTA: the late launch waves weigh more
        for ws, wtab in ((0, (1091, 1036, 964, 909)), (1, (1130, 1050, 950, 870))):
            for i in cutlass.range_constexpr(4):  # the rank for each possible wave (constant divisors), keep our own
                pw = base_p
                for j in cutlass.range_constexpr(4):
                    if cutlass.const_expr(j < i):
                        num = t2 * wtab[j] - wtab[i]  # >= 0 for every tid since wtab[j] > wtab[i]
                        pw += (num // (2 * wtab[i]) + 1) * sms_c
                    if cutlass.const_expr(j > i):
                        num = t2 * wtab[j] - wtab[i]
                        pw += (cutlass.max(num + 2 * wtab[i] - 1, Int32(0)) // (2 * wtab[i])) * sms_c
                if wave_i == i:
                    if strong == (ws == 1):
                        p = pw
    # The first due program is program bid whenever that one is due (thread 0's candidate): for small calls warps 1-3
    # read its fill counter themselves and send its window now instead of after the setup barriers.
    early = cutlass.Boolean(False)
    if cutlass.const_expr(fraction is not None):
        if warp != 0:
            if bid < nprog:
                s0 = ix[bid // 32]
                if s0 > 0:
                    early = hcnt[s0 * 32 + bid % 32] == 16
                if early:
                    issue_window(base, kb, gb, wb, pb, raw, s0 * 32 + bid % 32, mb_win.toint().to(Int32), mb_win, warp, lane, tma_atom_u, t_gu_e, t_su_e, fraction)
    slot0 = Int32(0)
    flag = cutlass.Boolean(False)
    if p < nprog:
        slot0 = ix[p // 32]
        if slot0 > 0:
            flag = hcnt[slot0 * 32 + p % 32] == 16
    # ---- due programs ranked in (warp, lane) order ----
    mask = cute.arch.vote_ballot_sync(flag)
    if lane == 0:
        counts[warp] = cute.arch.popc(mask)
    cute.arch.sync_threads()
    base_rank, total = Int32(0), Int32(0)
    count4 = lds128(base + CNT)
    for j in cutlass.range_constexpr(4):
        if j < warp:
            base_rank += count4[j]
        total += count4[j]
    rank = base_rank + cute.arch.popc(mask & ((cutlass.Uint32(1) << lane) - 1))
    if flag:
        due[rank] = slot0 * 32 + p % 32
    cute.arch.sync_threads()
    if cutlass.const_expr(fraction is not None):
        if warp != 0:
            if not early:
                if total > 0:
                    issue_window(base, kb, gb, wb, pb, raw, due[0], mb_win.toint().to(Int32), mb_win, warp, lane, tma_atom_u, t_gu_e, t_su_e, fraction)

    tp = cute.arch.retrieve_tmem_ptr(Float32, 16, tmh)
    tbase = tp.toint().to(Int32) + ((warp * 32) << 16)

    # ---- MMA atoms and operand views (all descriptors built by CuTe from the smem layouts) ----
    sw32b = cute.make_swizzle(1, 4, 3)
    def tile16(off, dtype):
        return cute.make_tensor(cute.make_ptr(dtype, base + off, cute.AddressSpace.smem, assumed_align=128, swizzle_=sw32b),
                                cute.make_layout((128, 16), stride=(16, 1)))
    # window products: A = ubuf exactly as the TMA lands it (MN-major SW128, j-atoms first), B = keys hi / lo
    mma_rb = cute.make_tiled_mma(tc.MmaF16BF16Op(BFloat16, Float32, (128, 128, 16), tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                                                 cute.nvgpu.OperandMajorMode.MN, cute.nvgpu.OperandMajorMode.K))
    thr_rb = mma_rb.get_slice(0)
    ub_layout = cute.tile_to_shape(tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.MN_SW128, BFloat16), (128, 16), order=(1, 0))
    ub_tile = cute.make_tensor(cute.make_ptr(BFloat16, base + UB, cute.AddressSpace.smem, assumed_align=1024, swizzle_=ub_layout.inner),
                               ub_layout.outer)
    a_win = mma_rb.make_fragment_A(thr_rb.partition_A(ub_tile))
    b_hi = mma_rb.make_fragment_B(thr_rb.partition_B(tile16(KB, BFloat16)))
    b_lo = mma_rb.make_fragment_B(thr_rb.partition_B(tile16(GB, BFloat16)))
    g_u = cute.local_tile(tma_u, (128, 16), (0, 0, None))
    t_su, t_gu = cpasync.tma_partition(tma_atom_u, 0, cute.make_layout(1), cute.group_modes(ub_tile, 0, 2), cute.group_modes(g_u, 0, 2))
    frag = mma_rb.make_fragment_C(mma_rb.partition_shape_C((128, 128)))
    acc_t = cute.make_tensor(tp, frag.layout)

    codes_layout = cute.tile_to_shape(tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW128, Int8), (128, 128), order=(0, 1))
    codes_in = cute.make_tensor(cute.make_ptr(Int8, base + SF + 16384, cute.AddressSpace.smem, assumed_align=1024, swizzle_=codes_layout.inner),
                                codes_layout.outer)
    g_c = cute.local_tile(tma_c, (128, 128), (0, 0, None, None))
    t_sc, t_gc = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), cute.group_modes(codes_in, 0, 2), cute.group_modes(g_c, 0, 2))
    codes_out = cute.make_tensor(cute.make_ptr(Int8, base + SF, cute.AddressSpace.smem, assumed_align=1024, swizzle_=codes_layout.inner),
                                 codes_layout.outer)
    g_s = cute.local_tile(tma_s, (128, 128), (0, 0, None, None))
    t_ss, t_gs = cpasync.tma_partition(tma_atom_s, 0, cute.make_layout(1), cute.group_modes(codes_out, 0, 2), cute.group_modes(g_s, 0, 2))
    state_atom = tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW128, Float16)
    state_layout = cute.tile_to_shape(state_atom, (128, 128), order=(0, 1))
    state = cute.make_tensor(cute.make_ptr(Float16, base + SF, cute.AddressSpace.smem, assumed_align=1024, swizzle_=state_layout.inner),
                             state_layout.outer)
    state_t = cute.make_tensor(state.iterator, cute.select(state.layout, mode=[1, 0]))
    src16 = cute.make_tensor(cute.make_ptr(Float16, base + SKB, cute.AddressSpace.smem, assumed_align=128), cute.make_layout((8, 128), stride=(1, 8)))
    mma_p = cute.make_tiled_mma(tc.MmaF16BF16Op(Float16, Float32, (128, 8, 16), tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                                                cute.nvgpu.OperandMajorMode.MN, cute.nvgpu.OperandMajorMode.MN))
    mma_q = cute.make_tiled_mma(tc.MmaF16BF16Op(Float16, Float32, (128, 8, 16), tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                                                cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN))
    thp, thq = mma_p.get_slice(0), mma_q.get_slice(0)
    ps_a = mma_p.make_fragment_A(thp.partition_A(state_t))
    ps_b = mma_p.make_fragment_B(thp.partition_B(src16))
    seed16 = cute.make_tensor(cute.make_ptr(Float16, base + SEEDB, cute.AddressSpace.smem, assumed_align=128), cute.make_layout((8, 128), stride=(1, 8)))
    ps_seed = mma_p.make_fragment_B(thp.partition_B(seed16))
    qs_a = mma_q.make_fragment_A(thq.partition_A(state))
    qs_b = mma_q.make_fragment_B(thq.partition_B(src16))
    skinny_frag = mma_p.make_fragment_C(mma_p.partition_shape_C((128, 8)))
    # the skinny products land in TMEM columns 120-127 (S' columns saved in `keep` from pass A's last chunk, so the
    # saved values are not live during pass A)
    skinny_acc = cute.make_tensor(cute.make_tensor(tp, skinny_frag.layout).iterator + Int32(120), skinny_frag.layout)
    res_layout = cute.tile_to_shape(tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW32, TFloat32), (128, 8), order=(0, 1))
    mma_r = cute.make_tiled_mma(tc.MmaTF32Op((128, 128, 8), tc.CtaGroup.ONE, tc.OperandSource.SMEM,
                                             cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K))
    thr_r = mma_r.get_slice(0)
    ra = cute.make_tensor(cute.make_ptr(TFloat32, base + SF, cute.AddressSpace.smem, assumed_align=1024, swizzle_=res_layout.inner), res_layout.outer)
    rb = cute.make_tensor(cute.make_ptr(TFloat32, base + SF + 4096, cute.AddressSpace.smem, assumed_align=1024, swizzle_=res_layout.inner), res_layout.outer)
    r_a = mma_r.make_fragment_A(thr_r.partition_A(ra))
    r_b = mma_r.make_fragment_B(thr_r.partition_B(rb))

    if total > 0:
        if tid == 0:
            cute.arch.mbarrier_init(mb_mma, 1)
            if cutlass.const_expr(fraction is None):
                cute.arch.mbarrier_init(mb_win, 3)
            cute.arch.mbarrier_init(mb_codes, 1)
            cute.arch.mbarrier_init_fence()
        zv = cute.make_rmem_tensor(2, Int32)       # columns 4..7 of the skinny operand stay zero
        zv[0], zv[1] = Int32(0), Int32(0)
        cute.autovec_copy(zv, cute.make_tensor(cute.make_ptr(Int32, base + SKB + tid * 16 + 8, cute.AddressSpace.smem, assumed_align=8), cute.make_layout(2)))
        ov = cute.make_rmem_tensor(2, Int32)
        ov[0], ov[1] = Int32(0x3c003c00), Int32(0x3c003c00)
        cute.autovec_copy(ov, cute.make_tensor(cute.make_ptr(Int32, base + ONES + tid * 8, cute.AddressSpace.smem, assumed_align=8), cute.make_layout(2)))
        fence_async_smem()
        cute.arch.sync_threads()
    if total > 0:
        if cutlass.const_expr(fraction is None):
            if warp != 0:
                issue_window(base, kb, gb, wb, pb, raw, due[0], mb_win.toint().to(Int32), mb_win, warp, lane, tma_atom_u, t_gu, t_su, fraction)
            zseed = seed_load(start, tid)
        else:
            zseed = zseed_e
        # conditioned start basis: power-of-two column scaling, then CholeskyQR2 (QVB is free until the Q stage)
        seed_scale(zseed, base + QVB, tid)
        orthogonalize(base + QVB, base + SEEDB, tid, Float32(TAU), True)
    for work in range(total):
        ph = due[work]
        slot, head = ph // 32, ph % 32
        rawhead = raw + (Int64(OFF) + slot.to(Int64) * SS + head.to(Int64) * 16384)
        mbar_wait(mb_win, work & 1)
        if work == 0:                     # the first codes follow the first window (the initial DRAM burst serves windows first)
            if tid == 0:
                mbar_expect(mb_codes.toint().to(Int32), 16384)
            if warp == 0:
                cute.copy(tma_atom_c, t_gc[(None, head, slot)], t_sc, tma_bar_ptr=mb_codes, cache_policy=l2_policy(fraction))
        # ---------------- prep part 1: everything into registers, bound partials ----------------
        # thread t prepares key pi(t), so that its operand rows (TMEM column t) are stored conflict-free
        kcol = pi_of(tid)
        pcv = cute.math.exp(cute.make_tensor(cute.make_ptr(Float32, base + PCM, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(128))[kcol])
        meta = cute.make_tensor(cute.make_ptr(Float32, base + META, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(256))
        skv = meta[kcol] * pcv
        svv = meta[128 + tid]
        wv = cute.make_rmem_tensor(16, Float32)
        for q in cutlass.range_constexpr(4):
            w4 = lds128f(base + WGT + q * 16)
            for i in cutlass.range_constexpr(4):
                wv[q * 4 + i] = w4[i]
        keys = cute.make_rmem_tensor(16, Float32)
        prod = Float32(1)
        kmax = Float32(0)
        for jj in cutlass.range_constexpr(16):
            j = 15 - jj
            kbits = lds_u16(base + KB + j * 256 + kcol * 2)
            gbits = lds_u16(base + GB + j * 256 + kcol * 2)
            kf = bits_float(kbits << 16)
            keys[j] = kf * wv[j] * prod
            kmax = cute.arch.fmax(kmax, cute.math.abs(keys[j]))
            gh = cute.make_rmem_tensor(1, Int32)
            gh[0] = gbits
            prod = prod * (cute.recast_tensor(gh, Float16)[0]).to(Float32)
        uw8 = lds128(base + UB + tid * 16) + lds128(base + UB + 2048 + tid * 16)   # any 16 values: only the tile maximum
        um = uw8[0] & Int32(0x7FFF7FFF)
        for i in cutlass.range_constexpr(1, 8):
            um = max_u16x2(um, uw8[i] & Int32(0x7FFF7FFF))
        umax = cute.arch.fmax(bits_float(um << 16), bits_float(um & Int32(-65536)))
        uqw = cute.make_rmem_tensor(8, Int32)      # u_old[r][kcol] (r 0..3) and q_old[r][tid]
        for r in cutlass.range_constexpr(4):
            uqw[r] = lds_u16(base + META + 1024 + r * 512 + kcol * 2)
            uqw[4 + r] = lds_u16(base + META + 1024 + r * 512 + 256 + tid * 2)
        uo = cute.make_rmem_tensor(4, Float32)
        qo = cute.make_rmem_tensor(4, Float32)
        qmax, uomax = Float32(0), Float32(0)
        for r in cutlass.range_constexpr(4):
            hu = cute.make_rmem_tensor(2, Int32)
            hu[0], hu[1] = uqw[r], uqw[4 + r]
            hv = cute.recast_tensor(hu, Float16)
            uo[r] = hv[0].to(Float32) * pcv
            qo[r] = hv[2].to(Float32)
            uomax = cute.arch.fmax(uomax, cute.math.abs(uo[r]))
            qmax = cute.arch.fmax(qmax, cute.math.abs(qo[r]))
        parts = (cute.math.abs(skv), cute.math.abs(svv), qmax, uomax, umax, kmax)
        if warp == 0:
            bulk_wait_read()          # the previous program's code store no longer reads SF (warp 0 issued it)
        for j in cutlass.range_constexpr(6):
            m = redux_max(parts[j])
            if lane == 0:
                bnd[warp, j] = m
        cute.arch.sync_threads()
        mx = cute.make_rmem_tensor(6, Float32)
        for j in cutlass.range_constexpr(6):
            mx[j] = cute.arch.fmax(cute.arch.fmax(bnd[0, j], bnd[1, j]), cute.arch.fmax(bnd[2, j], bnd[3, j]))
        residual_bound = mx[0] * mx[1]
        update_bound = Float32(4) * mx[2] * mx[3] + Float32(16) * mx[4] * mx[5]
        bound = mx[0] * mx[1] + Float32(4) * mx[2] * mx[3] + Float32(16) * mx[4] * mx[5]
        # Positive scales can remain on an empty old residual. If their bound
        # overwhelms the factors and window, it can underflow the fp16 state.
        # Remove that contribution only after confirming zero codes on the GPU.
        if residual_bound > update_bound * Float32(65536):
            mbar_wait(mb_codes, work & 1)
            nz = Int32(0)
            for cc in cutlass.range_constexpr(8):
                cv = lds128(base + SF + 16384 + tid * 16 + cc * 2048)
                for jj in cutlass.range_constexpr(4):
                    nz = nz | cv[jj]
            vote = cute.arch.vote_ballot_sync(nz != Int32(0))
            if lane == 0:
                counts[warp] = vote.to(Int32)
            cute.arch.sync_threads()
            if (counts[0] | counts[1] | counts[2] | counts[3]) == Int32(0):
                bound = update_bound
                skv = Float32(0)  # confirmed zero codes: avoid 0 * infinity after scaling
        bexp = ((float_bits(bound) >> 23) & 255) - 127
        e = cutlass.min(cutlass.max((Int32(13) - bexp) & Int32(-2), Int32(-120)), Int32(120))
        # Keep the scaled key and old-U operands finite even if their matching A operand is zero.
        operand_peak = cute.arch.fmax(mx[3], mx[5])
        operand_cap = (Int32(247) - ((float_bits(operand_peak) >> 23) & 255)) & Int32(-2)
        e = cutlass.min(e, operand_cap)
        up = bits_float((e + 127) << 23)
        half_down = bits_float((127 - (e >> 1)) << 23)
        # ---------------- prep part 2: scaled operands ----------------
        ncol = tid
        skp[ncol] = skv * Float32(1.0 / 127.0) * up
        kh = cute.make_rmem_tensor(8, Int32)
        kl = cute.make_rmem_tensor(8, Int32)
        for jp in cutlass.range_constexpr(8):
            a0, a1 = cute.arch.mul_packed_f32x2((keys[2 * jp], keys[2 * jp + 1]), (up, up), ftz=False)
            h = pack_bf16(a0, a1)
            r0 = a0 - bits_float(h << 16)
            r1 = a1 - bits_float(h & Int32(-65536))
            kh[jp] = h
            kl[jp] = pack_bf16(r0, r1)
        for c in cutlass.range_constexpr(2):
            sts128(sw32(base + KB, ncol, c), kh[4 * c], kh[4 * c + 1], kh[4 * c + 2], kh[4 * c + 3])
            sts128(sw32(base + GB, ncol, c), kl[4 * c], kl[4 * c + 1], kl[4 * c + 2], kl[4 * c + 3])
        # fp16 Q is exact in TF32; split scaled, decayed U into two TF32 terms.
        qrow = cute.make_rmem_tensor(4, Float32)
        uhi = cute.make_rmem_tensor(4, Int32)
        ulo = cute.make_rmem_tensor(4, Int32)
        for r in cutlass.range_constexpr(4):
            qrow[r] = qo[r]
            us = uo[r] * up
            uhi[r] = tf32_bits(us)
            ulo[r] = tf32_bits(us - bits_float(uhi[r]))
        for c in cutlass.range_constexpr(2):
            cute.autovec_copy(qrow, cute.make_tensor(cute.make_ptr(Float32, sw32(base + SF, tid, c), cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
            usrc = uhi if c == 0 else ulo
            cute.autovec_copy(usrc, cute.make_tensor(cute.make_ptr(Int32, sw32(base + SF + 4096, ncol, c), cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
        fence_async_smem()
        cute.arch.sync_threads()
        if warp == 0:
            mma_rb.set(tc.Field.ACCUMULATE, False)
            cute.gemm(mma_rb, acc_t, a_win, b_hi, acc_t)
            mma_rb.set(tc.Field.ACCUMULATE, True)
            cute.gemm(mma_rb, acc_t, a_win, b_lo, acc_t)
            mma_r.set(tc.Field.ACCUMULATE, True)
            cute.gemm(mma_r, acc_t, r_a, r_b, acc_t)
            with cute.arch.elect_one():
                tc.commit(mb_mma, None, tc.CtaGroup.ONE)
        # window resets (the inputs are in smem; issued while the rebuild product runs; for small calls in the P wait,
        # after the codes' DRAM burst)
        if cutlass.const_expr(fraction is None):
            reset_window(base, gb, wb, pb, ph, tid, fraction)
        mbar_spin(mb_mma.toint().to(Int32), work & 1)
        tc_after()
        mbar_wait(mb_codes, work & 1)
        # ---------------- pass A: S' = acc + c * skpc' * sv (thread = row), back to TMEM, fp16 copy ----------------
        r8 = tid & 7
        keep = cute.make_rmem_tensor(8, Float32)
        # 16-column chunks, software-pipelined: chunk h+1's TMEM values and codes are in flight while chunk h is
        # converted (each thread reads and rewrites only its own row; chunk h's fp16 stores reach only code
        # chunks that are already consumed)
        acc_n = tmem_load_nowait(tbase, '32x32b.x16', 16)
        cw_n = lds128(sw128(base + SF + 16384, tid, 0))
        tc_wait_ld()
        for h in cutlass.range_constexpr(8):
            j, g = h // 2, h % 2
            acc, cw = acc_n, cw_n
            if cutlass.const_expr(h < 7):
                acc_n = tmem_load_nowait(tbase + 16 * (h + 1), '32x32b.x16', 16)
                cw_n = lds128(sw128(base + SF + 16384, tid, h + 1))
            sk16 = []
            for q in cutlass.range_constexpr(4):
                sk16 += lds128f(base + SKP + (16 * h + 4 * q) * 4)
            out = [None] * 16
            for hh in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(4):
                    n0 = 8 * hh + 2 * c
                    word = cw[c]
                    # Exact byte-to-float conversion through the fp32 mantissa,
                    # keeping both bytes on the packed arithmetic path.
                    wx = word ^ Int32(-2139062144)
                    c0, c1 = cute.arch.add_packed_f32x2((magic_byte(wx, 2 * hh), magic_byte(wx, 2 * hh + 1)), (Float32(-32896.0), Float32(-32896.0)))
                    d0, d1 = cute.arch.mul_packed_f32x2((c0, c1), (sk16[n0], sk16[n0 + 1]), ftz=False)
                    out[n0], out[n0 + 1] = cute.arch.fma_packed_f32x2((d0, d1), (svv, svv), (acc[n0], acc[n0 + 1]), ftz=False)
            tmem_store(tbase + 16 * h, '32x32b.x16', out)
            if cutlass.const_expr(h == 7):
                for i in cutlass.range_constexpr(8):
                    keep[i] = out[8 + i]
            hw = [pack_half(out[2 * i], out[2 * i + 1]) for i in range(8)]
            for qq in cutlass.range_constexpr(2):
                q = 2 * g + qq
                sts128(base + SF + (j // 2) * 16384 + tid * 128 + (((4 * (j % 2) + q) ^ r8) << 4), hw[4 * qq], hw[4 * qq + 1], hw[4 * qq + 2], hw[4 * qq + 3])
            if cutlass.const_expr(h < 7):
                tc_wait_ld()
        if cutlass.const_expr(dbg == 1):
            for j in cutlass.range_constexpr(4):
                vals = tmem_load(tbase + 32 * j, '32x32b.x32', 32)
                for i in cutlass.range_constexpr(32):
                    dbgp[(ph * 128 + tid) * 129 + 32 * j + i] = vals[i]
            dbgp[(ph * 128 + tid) * 129 + 128] = up
        tc_wait_st()
        fence_async_smem()
        tc_before()
        cute.arch.sync_threads()
        # ---------------- subspace iteration: P = S'^T q0, Q = S' P, QR, U = S'^T Q ----------------
        if warp == 0:
            tc_after()
            mma_p.set(tc.Field.ACCUMULATE, False)
            for it in cutlass.range_constexpr(8):
                cute.gemm(mma_p, skinny_acc, ps_a[None, None, it], ps_seed[None, None, it], skinny_acc)
                mma_p.set(tc.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tc.commit(mb_mma, None, tc.CtaGroup.ONE)
        if cutlass.const_expr(fraction is not None):
            reset_window(base, gb, wb, pb, ph, tid, fraction)
        mbar_spin(mb_mma.toint().to(Int32), (work & 1) ^ 1)
        tc_after()
        pv8 = tmem_load(tbase + 120, '32x32b.x8', 8)        # S'^T (q0 high) and S'^T (q0 low)
        pv = [None] * 4
        for r in cutlass.range_constexpr(2):
            pv[2*r], pv[2*r+1] = cute.arch.add_packed_f32x2((pv8[2*r],pv8[2*r+1]),(pv8[4+2*r],pv8[5+2*r]))
        psc = [None] * 4
        for rr in cutlass.range_constexpr(2):
            psc[2*rr], psc[2*rr+1] = cute.arch.mul_packed_f32x2((pv[2*rr],pv[2*rr+1]),(Float32(1.0/256),Float32(1.0/256)),ftz=False)
        pw2 = cute.make_rmem_tensor(2, Int32)
        pw2[0], pw2[1] = pack_half(psc[0],psc[1]), pack_half(psc[2],psc[3])
        cute.autovec_copy(pw2, cute.make_tensor(cute.make_ptr(Int32, base + SKB + tid * 16, cute.AddressSpace.smem, assumed_align=8), cute.make_layout(2)))
        fence_async_smem()
        tc_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_after()
            mma_q.set(tc.Field.ACCUMULATE, False)
            for it in cutlass.range_constexpr(8):
                cute.gemm(mma_q, skinny_acc, qs_a[None, None, it], qs_b[None, None, it], skinny_acc)
                mma_q.set(tc.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tc.commit(mb_mma, None, tc.CtaGroup.ONE)
        # the next window (its regions are free since the rebuild product) while Q runs: issued in the P wait, its
        # copies kept warps 0 / 1 ~300 cycles behind the others at the P-stage barrier
        if warp != 0:
            if work + 1 < total:
                issue_window(base, kb, gb, wb, pb, raw, due[work + 1], mb_win.toint().to(Int32), mb_win, warp, lane, tma_atom_u, t_gu, t_su, fraction)
        mbar_spin(mb_mma.toint().to(Int32), work & 1)
        tc_after()
        qv4 = tmem_load(tbase + 120, '32x32b.x4', 4)      # (the skinny operand's columns 4-7 are zero)
        qrow = cute.make_rmem_tensor(4, Float32)
        for r in cutlass.range_constexpr(4):
            qrow[r] = qv4[r]
        cute.autovec_copy(qrow, cute.make_tensor(cute.make_ptr(Float32, base + QVB + tid * 16, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
        tc_before()
        cute.arch.sync_threads()
        orthogonalize(base + QVB, base + SKB, tid, Float32(TAU), False)
        # U = S'^T Q: only warp 0 wrote Q (the fp16 state and the skinny accumulator columns are ordered by the
        # earlier barriers), so warp 0 issues the product right after its QR; the barrier only orders warps 1-3's
        # reads of their Q rows in the U stage
        if warp == 0:
            fence_async_smem()
            cute.arch.sync_warp()          # every lane's Q rows (and fence) before the elected lane's MMA issue
            tc_after()
            mma_p.set(tc.Field.ACCUMULATE, False)
            for it in cutlass.range_constexpr(8):
                cute.gemm(mma_p, skinny_acc, ps_a[None, None, it], ps_b[None, None, it], skinny_acc)
                mma_p.set(tc.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tc.commit(mb_mma, None, tc.CtaGroup.ONE)
        cute.arch.sync_threads()
        mbar_spin(mb_mma.toint().to(Int32), (work & 1) ^ 1)
        tc_after()
        uv = tmem_load(tbase + 120, '32x32b.x4', 4)       # (the skinny operand's columns 4-7 are zero)
        tmem_store(tbase + 120, '32x32b.x8', [keep[i] for i in range(8)])     # restore S' columns 120-127 (U is read)
        # new factors: u_new = fp16(U' 2^-e) at key pi(tid), q_new = fp16 Q row tid
        down = bits_float((127 - e) << 23)
        uq = cute.make_tensor(cute.recast_ptr(rawhead + 4352, dtype=Float16), cute.make_layout((4, 256), stride=(256, 1)))
        kq = pi_of(tid)
        qh = cute.make_rmem_tensor(2, Int32)
        cute.autovec_copy(cute.make_tensor(cute.make_ptr(Int32, base + SKB + tid * 16, cute.AddressSpace.smem, assumed_align=8), cute.make_layout(2)), qh)
        qhh = cute.recast_tensor(qh, Float16)
        rbw = cute.make_rmem_tensor(4, Float32)
        raw4 = cute.make_rmem_tensor(4, Float32)
        uns = []
        for r in cutlass.range_constexpr(4):
            un = (uv[r] * down).to(Float16)
            uns.append(un)
            rbw[r] = -(un.to(Float32) * up)
            raw4[r] = qhh[r].to(Float32)
        zero4 = cute.make_rmem_tensor(4, Float32)
        for r in cutlass.range_constexpr(4):
            zero4[r] = Float32(0)
        # residual operands (tf32, K-major SW32): A row tid = q_new, B row tid = -2^e u_new
        # (A's upper half keeps fp16 state bytes, always finite as tf32, and meets B's zero half)
        cute.autovec_copy(raw4, cute.make_tensor(cute.make_ptr(Float32, sw32(base + SF, tid, 0), cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
        for c in cutlass.range_constexpr(2):
            src_b = rbw if c == 0 else zero4
            cute.autovec_copy(src_b, cute.make_tensor(cute.make_ptr(Float32, sw32(base + SF + 4096, tid, c), cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
        tc_wait_st()
        fence_async_smem()
        tc_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_after()
            mma_r.set(tc.Field.ACCUMULATE, True)
            cute.gemm(mma_r, acc_t, r_a, r_b, acc_t)
            with cute.arch.elect_one():
                tc.commit(mb_mma, None, tc.CtaGroup.ONE)
        # the fp16 state is no longer read (U is done): the next program's codes may land in its upper half
        if warp == 3:                     # (warp 0 issues the products)
            if work + 1 < total:
                pn = due[work + 1]
                if lane == 0:
                    mbar_expect(mb_codes.toint().to(Int32), 16384)
                cute.copy(tma_atom_c, t_gc[(None, pn % 32, pn // 32)], t_sc, tma_bar_ptr=mb_codes, cache_policy=l2_policy(fraction))
        # the new factors leave while the residual product runs
        for r in cutlass.range_constexpr(4):
            uq[r, kq] = uns[r]
            uq[r, 128 + tid] = qhh[r]
        mbar_spin(mb_mma.toint().to(Int32), work & 1)
        tc_after()
        if cutlass.const_expr(dbg == 2):
            for j in cutlass.range_constexpr(4):
                vals = tmem_load(tbase + 32 * j, '32x32b.x32', 32)
                for i in cutlass.range_constexpr(32):
                    dbgp[(ph * 128 + tid) * 129 + 32 * j + i] = vals[i]
            dbgp[(ph * 128 + tid) * 129 + 128] = up
        # ---------------- pass B: column sums of |E'| in registers (16x256b, transpose-reduce over 8 lanes) ----------------
        cs = [None] * 32
        for lb in cutlass.range_constexpr(2):
            ev = tmem_load(tbase + ((16 * lb) << 16), '16x256b.x16', 64)
            for b in cutlass.range_constexpr(16):
                p0, p1 = cute.arch.add_packed_f32x2((cute.math.abs(ev[4 * b]), cute.math.abs(ev[4 * b + 1])),
                                                    (cute.math.abs(ev[4 * b + 2]), cute.math.abs(ev[4 * b + 3])))
                if cutlass.const_expr(lb == 0):
                    cs[2 * b], cs[2 * b + 1] = p0, p1
                else:
                    cs[2 * b], cs[2 * b + 1] = cute.arch.add_packed_f32x2((cs[2 * b], cs[2 * b + 1]), (p0, p1))
        cur = cs
        for st in cutlass.range_constexpr(3):
            half = 16 >> st
            hb = ((lane >> (4 - st)) & 1) == 1
            nxt = []
            for j in cutlass.range_constexpr(half):
                snd = Float32(cutlass.select_(hb, cur[j], cur[half + j]))
                kp = Float32(cutlass.select_(hb, cur[half + j], cur[j]))
                nxt.append(kp + cute.arch.shuffle_sync_bfly(snd, half))
            cur = nxt
        colp = cute.make_rmem_tensor(4, Float32)
        for j in cutlass.range_constexpr(4):
            colp[j] = cur[j]
        cute.autovec_copy(colp, cute.make_tensor(cute.make_ptr(Float32, base + QVB + warp * 512 + lane * 16, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(4)))
        cute.arch.sync_threads()
        colsum = cute.make_tensor(cute.make_ptr(Float32, base + QVB, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(512))
        csum = (colsum[tid] + colsum[128 + tid]) + (colsum[256 + tid] + colsum[384 + tid])
        half_up = bits_float((127 + (e >> 1)) << 23)
        rskn = cute.arch.fmin(cute.math.rsqrt(csum * Float32(1.0 / 128), fastmath=True), Float32(1e12) * half_down)
        scales = cute.make_tensor(rawhead + 4096, cute.make_layout(256))
        # Round the reciprocal once, then use its inverse as the output scale.
        # Keeping both sides consistent preserves the quantized reconstruction.
        # fp16 reciprocal (11-bit mantissa keeps the quantization grid within 2^-12 of the reference's, so the codes
        # track the reference's rounding; rskn >= 2^-7 here, the clamp only touches columns that are zero to ~1e-14
        # of the head), and its inverse as the output scale
        recip_h = cute.arch.fmin(rskn, Float32(65504.0)).to(Float16)
        recip = recip_h.to(Float32)
        scales[tid] = cute.arch.rcp_approx(recip) * half_down
        nt = pi_inv(tid)
        packed_sk = cute.make_tensor(cute.make_ptr(Float16, base + SKP, cute.AddressSpace.smem, assumed_align=16), cute.make_layout(128))
        packed_sk[(((nt >> 4) * 4 + ((nt >> 1) & 3)) * 4) + ((nt >> 3) & 1) * 2 + (nt & 1)] = recip_h
        ev_first = tmem_load_nowait(tbase, '16x256b.x16', 64)       # pass C's first block, in flight across the barrier
        cute.arch.sync_threads()
        tc_wait_ld()
        # ---------------- pass C: row scales and codes (16x256b, permuted columns), stmatrix into SF[0:16K] ----------------
        c4 = lane & 3
        eps_v = Float32(1e-12) * half_up
        rowsc = []
        rsk_packed = []
        for q in cutlass.range_constexpr(8):
            rsk_packed += _smem_vec(Int32, base + SKP + (q * 4 + c4) * 8, 2, 8)
        for lb in cutlass.range_constexpr(2):
            ev = ev_first if lb == 0 else tmem_load(tbase + ((16 * lb) << 16), '16x256b.x16', 64)
            x = [None] * 64
            m0, m1 = Float32(0), Float32(0)
            for b in cutlass.range_constexpr(16):
                rs = f16x2_f32(rsk_packed[b])
                x[4 * b], x[4 * b + 1] = cute.arch.mul_packed_f32x2((ev[4 * b], ev[4 * b + 1]), (rs[0], rs[1]), ftz=False)
                x[4 * b + 2], x[4 * b + 3] = cute.arch.mul_packed_f32x2((ev[4 * b + 2], ev[4 * b + 3]), (rs[0], rs[1]), ftz=False)
                m0 = cute.arch.fmax(m0, cute.arch.fmax(cute.math.abs(x[4 * b]), cute.math.abs(x[4 * b + 1])))
                m1 = cute.arch.fmax(m1, cute.arch.fmax(cute.math.abs(x[4 * b + 2]), cute.math.abs(x[4 * b + 3])))
            for sh in cutlass.range_constexpr(2):
                m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, 1 << sh))
                m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, 1 << sh))
            m0 = cute.arch.fmax(m0, eps_v)
            m1 = cute.arch.fmax(m1, eps_v)
            row0 = warp * 32 + 16 * lb + (lane >> 2)
            rowsc.append(m0 * half_down)
            rowsc.append(m1 * half_down)
            f0 = Float32(127) * cute.arch.rcp_approx(m0)
            f1 = Float32(127) * cute.arch.rcp_approx(m1)
            wds = [[None] * 8, [None] * 8]
            for mm in cutlass.range_constexpr(8):
                wds[0][mm] = quant_pack(x[8 * mm], x[8 * mm + 1], x[8 * mm + 4], x[8 * mm + 5], f0)
                wds[1][mm] = quant_pack(x[8 * mm + 2], x[8 * mm + 3], x[8 * mm + 6], x[8 * mm + 7], f1)
            jj = lane >> 3
            xr = jj & 1
            rho = lane & 7
            vrow = warp * 32 + 16 * lb + 8 * xr + rho
            for g in cutlass.range_constexpr(4):
                mcol = 2 * g + (jj >> 1)
                addr = base + SF + vrow * 128 + ((mcol ^ (vrow & 7)) << 4)
                stsm4(addr, wds[0][2 * g], wds[1][2 * g], wds[0][2 * g + 1], wds[1][2 * g + 1])
        fence_async_smem()
        cute.arch.sync_threads()
        if warp == 0:
            cute.copy(tma_atom_s, t_ss, t_gs[(None, head, slot)], cache_policy=l2_policy(fraction))
            bulk_commit()
        if c4 == 0:                   # row scales leave after the code store is issued
            for lb in cutlass.range_constexpr(2):
                row0 = warp * 32 + 16 * lb + (lane >> 2)
                scales[128 + row0] = rowsc[2 * lb]
                scales[136 + row0] = rowsc[2 * lb + 1]

    if total > 0:
        if warp == 0:
            bulk_wait_read()              # the last code store has read SF (its global writes complete with the grid)
        cute.arch.sync_threads()
    if warp == 0:
        cute.arch.dealloc_tmem(tp, 128)


@cute.jit
def launch(raw: cute.Pointer, kb: cute.Pointer, ub: cute.Pointer, gb: cute.Pointer, wb: cute.Pointer, pb: cute.Pointer,
           hc: cute.Pointer, desc: cute.Pointer, start: cute.Pointer, indices: cute.Pointer, dbgp: cute.Pointer,
           n: Int32, nprog_pool: Int32, stream: cuda.CUstream, blocks: cutlass.Constexpr, dbg: cutlass.Constexpr, fraction: cutlass.Constexpr = None):
    # the pool's codes as (row, code, head, slot) int8; one program's 128 x 128 tile lands K-major SW128 (descriptor
    # in the kernel parameters)
    codes_ptr = cute.make_ptr(Int8, raw.toint() + OFF * 4, cute.AddressSpace.gmem, assumed_align=16)
    codes_g = cute.make_tensor(codes_ptr, cute.make_layout((128, 128, 32, nprog_pool // 32), stride=(128, 1, 65536, 4 * SS)))
    codes_layout = cute.tile_to_shape(tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.K_SW128, Int8), (128, 128), order=(0, 1))
    tma_atom_c, tma_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), codes_g, codes_layout, (128, 128))
    tma_atom_s, tma_s = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), codes_g, codes_layout, (128, 128))
    # each program's ubuf as (value, entry, program), landing as the MN-major SW128 window A operand
    ub_g = cute.make_tensor(cute.make_ptr(BFloat16, ub.toint(), cute.AddressSpace.gmem, assumed_align=16),
                            cute.make_layout((128, 16, nprog_pool), stride=(1, 128, 2048)))
    ub_layout = cute.tile_to_shape(tc.make_smem_layout_atom(tc.SmemLayoutAtomKind.MN_SW128, BFloat16), (128, 16), order=(1, 0))
    tma_atom_u, tma_u = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), ub_g, ub_layout, (128, 16))
    flush_kernel(raw.toint(), kb.toint(), ub.toint(), gb.toint(), wb.toint(), pb.toint(),
                 hc.toint(), desc.toint(), start.toint(), indices.toint(), dbgp.toint(), n, tma_atom_c, tma_c,
                 tma_atom_s, tma_s, tma_atom_u, tma_u, blocks, dbg, fraction).launch(
        grid=(cutlass.max(Int32(1), cutlass.min(n * 32, Int32(blocks))), 1, 1),
        block=(128, 1, 1), min_blocks_per_mp=CTAS_PER_SM, stream=stream)


def _ptr(t, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


_COMPILED = {}


class Pool:
    """The serving engine's unchanged strided checkpoint and window layout."""
    def __init__(self, num_slots, device="cuda", dbg=0):
        self.NS = num_slots
        self.raw = torch.zeros(num_slots * SS + OFF, dtype=torch.float32, device=device)
        storage = self.raw.untyped_storage()
        def view(dtype, shape, strides, off):
            return torch.empty(0, dtype=dtype, device=device).set_(storage).as_strided(shape, strides, off)
        self.codes = view(torch.int8, (num_slots, 32, 128, 128), (4 * SS, 65536, 128, 1), 4 * OFF)
        self.codes32 = view(torch.int32, (num_slots, 32, 128, 32), (SS, 16384, 32, 1), OFF)
        self.codes16 = view(torch.int16, (num_slots, 32, 128, 64), (2 * SS, 32768, 64, 1), 2 * OFF)
        self.s_k = view(torch.float32, (num_slots, 32, 128), (SS, 16384, 1), OFF + 4096)
        self.s_v = view(torch.float32, (num_slots, 32, 128), (SS, 16384, 1), OFF + 4224)
        self.s_kv = view(torch.float32, (num_slots, 32, 256), (SS, 16384, 1), OFF + 4096)
        self.uq = view(torch.float16, (num_slots, 32, 4, 256), (2 * SS, 32768, 256, 1), 2 * (OFF + 4352))
        self.kbuf = torch.zeros(num_slots, 32, 16, 128, dtype=torch.bfloat16, device=device)
        self.ubuf = torch.zeros_like(self.kbuf)
        self.gbuf = torch.ones(num_slots, 32, 16, 128, dtype=torch.float16, device=device)
        self.w = torch.zeros(num_slots, 32, 16, dtype=torch.float32, device=device)
        self.pcum = torch.zeros(num_slots, 32, 128, dtype=torch.float32, device=device)
        self.hcnt = torch.zeros(num_slots, 32, dtype=torch.int32, device=device)
        self.dbg = dbg
        self.dbgbuf = torch.zeros(1 if not dbg else max(num_slots * 32 * 128 * 129, 2048 * 32 * 20), dtype=torch.float32, device=device)
        with torch.cuda.device(self.raw.device):
            self.codes_map = encode_map(self.codes.data_ptr(), (128, 128, 32, num_slots), (128, 65536, 4 * SS), (128, 128, 1, 1),
                                        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, self.raw.device)
        device_index = self.raw.device.index
        sms = torch.cuda.get_device_properties(self.raw.device).multi_processor_count
        self.blocks = sms * CTAS_PER_SM
        self._args = (_ptr(self.raw, Float32), _ptr(self.kbuf, BFloat16), _ptr(self.ubuf, BFloat16),
                      _ptr(self.gbuf, Float16), _ptr(self.w, Float32), _ptr(self.pcum, Float32), _ptr(self.hcnt, Int32),
                      _ptr(self.codes_map, Int64))
        self.kernels = {}
        with torch.cuda.device(device_index):
            # Every variant is ready before a caller captures flush_inplace.
            for fraction in (None, 0.75, 1.0):
                key = (device_index, self.blocks, dbg, fraction)
                if key not in _COMPILED:
                    compiled = cute.compile(launch, *self._args, _ptr(self.pcum, Float32), _ptr(self.hcnt, Int32), _ptr(self.dbgbuf, Float32),
                                            Int32(1), Int32(num_slots * 32), cuda.CUstream(0), self.blocks, dbg, fraction,
                                            options="--generate-line-info --ptxas-options=--maxrregcount=128")
                    _COMPILED[key] = compiled.to(device_index)
                self.kernels[fraction] = _COMPILED[key]
        self.kernel = self.kernels[None]


def flush_inplace(pool, idx, q0):
    if idx.numel() == 0:
        return
    with torch.cuda.device(pool.raw.device):
        count = idx.numel()
        kernel = pool.kernels[0.75 if count <= 20 else 1.0 if count <= 64 else None]
        kernel(*pool._args, _ptr(q0, Float32), _ptr(idx, Int32), _ptr(pool.dbgbuf, Float32), Int32(idx.numel()),
                    Int32(pool.NS * 32), cuda.CUstream(torch.cuda.current_stream(pool.raw.device).cuda_stream))


def run(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, q0):
    b = codes.shape[0]
    pool = Pool(b + 8, codes.device)
    sl = slice(1, b + 1)
    pool.codes[sl] = codes
    pool.s_k[sl] = s_k
    pool.s_v[sl] = s_v
    pool.uq[sl, :, :, :128] = u
    pool.uq[sl, :, :, 128:] = q
    pool.kbuf[sl], pool.ubuf[sl], pool.gbuf[sl] = kbuf, ubuf, gbuf
    pool.w[sl], pool.pcum[sl] = w, pcum
    pool.hcnt[sl] = 16
    idx = torch.arange(1, b + 1, dtype=torch.int32, device=codes.device)
    flush_inplace(pool, idx, q0.contiguous())
    return (pool.codes[sl].clone(), pool.s_k[sl].clone(), pool.s_v[sl].clone(),
            pool.uq[sl, :, :, :128].clone(), pool.uq[sl, :, :, 128:].clone())
