# Precision Analysis

The input/output formats and acceptance thresholds are unchanged. Internally,
FP32 state/residual retention has been added, while some lower-precision
approximations are used in projection, reduction, and scale computation. I have
rerun the original baseline and current implementation on B300 with exactly
the same inputs.

First, distinguish three things:

- **Mathematical reference:** mainly uses FP32 arithmetic, then outputs FP16
  low-rank factors and an INT8 residual.
- **Original TileLang baseline:** already uses mixed precision, including
  FP16/BF16 multiplication, FP32 accumulation, and FP16 intermediate storage.
- **Current CuTe implementation:** still uses mixed precision, but changes
  which intermediates retain FP32 and where approximations are allowed.

The external data types are completely unchanged:

| Data | Original baseline and current implementation |
|---|---|
| Residual `codes` | INT8, range [-127, 127] |
| Low-rank factors `u`, `q` | FP16 |
| Update buffers `kbuf`, `ubuf` | BF16 |
| Scales `s_k`, `s_v` | FP32 |
| `w`, `p`, `pcum`, `q0` | FP32 |
| KDA gate history `gbuf` | FP16 |

The numerical changes occur at the following stages.

1. **GDN state and residual: reduce FP16 rounding of matrix intermediates.**

   The original baseline first computes the quantized residual plus the window
   updates to obtain `B`, then stores `B` in FP16. The old rank-4 term participates
   separately through FP32 algebraic corrections. After the final residual is
   computed, it is stored in FP16 again before scales and codes are generated.

   The current implementation retains the full rebuilt state `S` in FP32 TMEM.
   It uses a scaled FP16 copy to extract the low-rank component, but computes

   $$
   E=S-q_{\rm new}^{T}u_{\rm new}
   $$

   using the retained FP32 state. The residual also remains in FP32 until
   quantization.

   This is the main precision improvement: when subtracting two large values
   produces a small residual, rounding the state to FP16 beforehand can make
   the rounding error large relative to that residual.

2. **KDA residual: FP32 reconstruction already existed; the current kernel
   removes the subsequent FP16 round trip.**

   The original KDA baseline enables `KEXACT=True` by default: it decodes the
   checkpoint again and recomputes the residual in FP32, avoiding direct use
   of the earlier FP16 state. However, it then scales the residual by a power
   of two per block and stores it in FP16.

   The current implementation retains the FP32 state directly, subtracts the
   new low-rank term from it, and keeps the residual in FP32 TMEM. The change
   therefore removes the FP16 store and reload after residual computation;
   it is not a change from an entirely FP16 pipeline to an entirely FP32 one.

3. **Product compensation: retain BF16 high/low splitting and reorganize the
   arithmetic.**

   The original baseline already uses BF16 high/low splitting. For an FP32
   value, the idea is

   $$
   x\approx x_{\rm hi}+x_{\rm lo},\qquad
   Ax\approx Ax_{\rm hi}+Ax_{\rm lo}.
   $$

   The current GDN implementation folds the weight into the key operand and
   then splits it into BF16 high and low parts; the baseline mainly folds
   the weight into the value operand. These are mathematically equivalent,
   but the rounding locations and accumulation order change.

   The current implementation also uses TF32 high/low compensation for
   weighted or decayed old low-rank terms. This retains more information
   than a single TF32 conversion, but still does not establish bitwise
   equivalence to IEEE FP32 computation.

4. **Starting matrix and dynamic range: add scaling and low-part protection.**

   The original baseline converts `q0` directly to FP16. The current
   implementation first scales each column by a power of two and retains
   FP16 high and low parts. KDA also conditions the starting basis through
   orthogonalization.

   The current implementation also chooses a per-head power-of-two scale
   during state reconstruction, keeping the FP16 projection copy within a
   suitable numerical range. The KDA baseline already had per-block scaling
   at the residual stage; the current kernel moves this protection earlier
   into the state stage.

   Scaling helps avoid overflow and underflow, but FP16 rounding remains.

5. **Orthogonalization: reduce computation according to numerical conditioning,
   while retaining corrections for difficult inputs.**

   The original baseline defaults to `GRAM16=False`, so the Gram matrices
   in both QR passes are computed in FP32.

   The current implementation skips the second orthogonalization pass when
   conditioning is good and adds corrections when conditioning is poor. GDN
   also uses state-low-part compensation and reorthogonalization for
   ill-conditioned cases. These choices change numerical results and are
   numerical tradeoffs.

   Also, `f32x2` in the code means computing two FP32 values simultaneously;
   it does not increase the precision of either value.

6. **Scales and reductions: these do introduce lower-precision approximations.**

   - **GDN:** part of the residual column summation uses packed FP16 addition
     before returning to FP32. The baseline reads the FP16 residual and
     accumulates those sums in FP32.
   - **KDA:** the column-scale reciprocal is rounded to FP16, then used to
     generate codes and the corresponding FP32 output scale. A scale stored
     in FP32 therefore need not have been computed entirely at FP32 precision.
   - Current row-scale normalization mainly uses approximate reciprocals;
     the baseline uses a correctly rounded reciprocal. GDN retains a more
     accurate fallback for extreme scales.
   - The final INT8 nearest-rounding rule and value range remain unchanged.

These paths can be compared directly in the original
[GDN baseline](gdn/baseline.py), [KDA baseline](kda/baseline.py), and current
[GDN](build/torch-cuda/_gdn.py) and [KDA](build/torch-cuda/_kda.py) implementations.

On the six given B300 workloads (batches 16, 32, 64, 128, 256, and 512), tested
with structured, second-window, and random inputs, both implementations pass
all 18 original cases per kernel. The worst reconstruction-error ratio relative
to the FP32 reference improves from **1.00324455 to 1.00041398 for GDN** and from
**1.00296773 to 1.00105573 for KDA**, with both errors measured against the same
FP64 rebuilt state. The current kernels' maximum individual-state
`MAE / mean(abs(S64))` is **0.576760% for GDN** and **0.612224% for KDA**.
An additional check of 96,768 individual states per kernel finds no GDN
violations and one KDA violation: a batch-128 random state has an error ratio
of **1.01256057**, exceeding the individual limit of 1.01. The original judge
averages over the batch dimension and still passes that case. Full measurements
are in the [GDN](gdn/precision_comparison.json) and
[KDA](kda/precision_comparison.json) reports.
