"""Benchmark GSW multiplication  C1 . G^{-1}(C2)  where the right factor is {0,1}.

Three regimes, so no slow path ever runs at full width:

  SMALL  (N=32)   : correctness of EVERY method vs an exact object-dtype
                    reference, plus the uint64 baseline timed (instant here).
                    This is the only place the slow uint64 / object matmuls run.

  MEDIUM (N=128)  : timed comparison of the BLAS methods and four Russians.
                    four Russians is Python gather/scatter (not BLAS) but takes
                    ~1 s here, so it is fine to time.  No uint64 (would be ~26 s),
                    no object reference.

  FULL   (N=512)  : a real LWE parameter.  ONLY BLAS-fast methods (single-float
                    and limb-split).  No uint64 (~28 min), no object ref (days),
                    no four Russians (minutes).  Correctness reference is the
                    limb-split path, cross-checked against single-float on the
                    {0,1} case (two independent code paths agreeing).

The methods:

  (a) SINGLE  :  float64(C1_centered) @ float64(B_centered) % q  -- one BLAS GEMM.
                 Valid because B is {0,1}: subset sums fit EXACTLY in float64's
                 53-bit mantissa (W * 2^qbits < 2^53).  The {0,1} observation's
                 concrete gift -- BLAS instead of numpy's ~8e7-ops/s uint64 path.

  (b) LIMB    :  split into 16-bit limbs, 4 BLAS GEMMs, recombine mod 2^qbits.
                 Works for FULL-RANGE @ FULL-RANGE too (each sub-product
                 < 2^16 * 2^16 * W = 2^46 < 2^53).  The scheme's decrypt, keygen,
                 and M.G matmuls use this route.

  (c) 4RUSS   :  method of four Russians -- block l into t-bit chunks, precompute
                 2^t subset sums per chunk (Gray-code DP, NO multiplications),
                 then each output entry is W/t table lookups.  O(N W^2 / log W):
                 the ASYMPTOTIC win the {0,1} structure predicts.  Pure subset-sum.

Run:  .venv/bin/python bench_mul.py
"""
import numpy, time

qbits = 32; q = 1 << qbits

def _to_signed(x):
    xi = x.astype(numpy.int64)
    return numpy.where(xi > q // 2, xi - q, xi)

def matmul_single(C1, B):
    As = _to_signed(C1); Bs = _to_signed(B)
    w = C1.shape[1]
    amax = int(numpy.abs(As).max()) if As.size else 0
    bmax = int(numpy.abs(Bs).max()) if Bs.size else 0
    assert amax * bmax * w < (1 << 53), "single-float not safe for this operand range"
    return numpy.uint64((numpy.float64(As) @ numpy.float64(Bs)) % q)

def matmul_limb(A, B, split=16):
    mask = (1 << split) - 1
    A_lo = numpy.float64(A & mask); A_hi = numpy.float64(A >> split)
    B_lo = numpy.float64(B & mask); B_hi = numpy.float64(B >> split)
    LL = numpy.uint64(A_lo @ B_lo)
    cross = (numpy.uint64(A_lo @ B_hi) + numpy.uint64(A_hi @ B_lo)) << split
    return (LL + cross) % q

def four_russians(C1, B, t, dtype=numpy.float64):
    N, W = C1.shape
    nblocks = (W + t - 1) // t
    Wpad = nblocks * t
    C1p = numpy.zeros((N, Wpad), dtype=dtype); C1p[:, :W] = C1.astype(dtype)
    Bp  = numpy.zeros((Wpad, W), dtype=numpy.uint64); Bp[:W, :] = B
    C1b = C1p.reshape(N, nblocks, t)
    Bb  = Bp.reshape(nblocks, t, W)
    pat = numpy.zeros((nblocks, W), dtype=numpy.int64)
    for b in range(t):
        pat += (Bb[:, b, :].astype(numpy.int64) << b)
    out = numpy.zeros((N, W), dtype=dtype)
    P = 1 << t
    ar = numpy.arange(P)
    for c in range(nblocks):
        Tc = numpy.zeros((N, P), dtype=dtype)
        for b in range(t):
            low = ar[((ar >> b) & 1) == 0]
            Tc[:, low | (1 << b)] = Tc[:, low] + C1b[:, c, b][:, None]
        out += Tc[:, pat[c]]
    return out

def ref_object(A, B):                              # exact, slow -- SMALL N only
    return (numpy.matmul(A.astype(object), B.astype(object)) % q).astype(numpy.uint64)

def timed(fn, reps=1):
    t0 = time.perf_counter(); r = None
    for _ in range(reps): r = fn()
    return (time.perf_counter() - t0) / reps, r

rng = numpy.random.default_rng(0)

# ---------------- SMALL: correctness vs exact object reference ----------------
print('=== SMALL (N=32, W=1024): correctness vs object-dtype reference ===')
N0 = 32; W0 = N0 * qbits
A0 = rng.integers(0, q, (N0, W0), dtype=numpy.uint64)
Bbin0 = rng.integers(0, 2, (W0, W0), dtype=numpy.uint64)        # {0,1} rhs (like G^{-1})
Bfull0 = rng.integers(0, q, (W0, W0), dtype=numpy.uint64)       # full-range rhs
r_ref_bin  = ref_object(A0, Bbin0)
r_ref_full = ref_object(A0, Bfull0)
dt_u64, _ = timed(lambda: (A0 @ Bbin0) % q)                     # uint64 baseline (instant at N=32)
print(f'  uint64 @         correct={numpy.array_equal((A0 @ Bbin0) % q, r_ref_bin)}   time={dt_u64*1e3:.2f}ms')
print(f'  single  {{0,1}}    correct={numpy.array_equal(matmul_single(A0, Bbin0), r_ref_bin)}')
print(f'  limb    {{0,1}}    correct={numpy.array_equal(matmul_limb(A0, Bbin0),   r_ref_bin)}')
print(f'  limb    full     correct={numpy.array_equal(matmul_limb(A0, Bfull0),  r_ref_full)}')
print(f'  4Russ t=4 {{0,1}}  correct={numpy.array_equal(numpy.uint64(four_russians(A0, Bbin0, 4)) % q, r_ref_bin)}')
print(f'  4Russ t=8 {{0,1}}  correct={numpy.array_equal(numpy.uint64(four_russians(A0, Bbin0, 8)) % q, r_ref_bin)}')

# ---------------- MEDIUM: timed comparison, no uint64 / no object ref ----------------
print()
print('=== MEDIUM (N=128, W=4096): timed comparison (no slow uint64/object paths) ===')
N1 = 128; W1 = N1 * qbits
A1 = rng.integers(0, q, (N1, W1), dtype=numpy.uint64)
Bbin1 = rng.integers(0, 2, (W1, W1), dtype=numpy.uint64)
r1_ref = matmul_limb(A1, Bbin1)                                 # limb as reference
dt_s1, r_s1 = timed(lambda: matmul_single(A1, Bbin1))
dt_l1, _    = timed(lambda: matmul_limb(A1, Bbin1))
print(f'  single float64 BLAS : {dt_s1:7.3f}s  correct={numpy.array_equal(r_s1, r1_ref)}  ({N1*W1*W1/dt_s1:.2e} ops/s)')
for t in [4, 8]:
    dt_c, r_c = timed(lambda t=t: four_russians(A1, Bbin1, t))
    ok = numpy.array_equal(numpy.uint64(r_c) % q, r1_ref)
    ops_c = (W1 // t) * N1 * (W1 + (1 << t))
    print(f'  4Russ t={t}          : {dt_c:7.3f}s  correct={ok}  ({ops_c/dt_c:.2e} ops/s, {N1*W1*W1/ops_c:.1f}x fewer ops than BLAS)')

# ---------------- FULL: BLAS-fast methods only, N=512 ----------------
print()
print('=== FULL (N=512, W=16384): BLAS-fast methods only (real LWE parameter) ===')
N = 512; W = N * qbits
print(f'  ct_mul ops = N*W^2 = {N*W*W:,}')
A  = rng.integers(0, q, (N, W), dtype=numpy.uint64)
Bbin = rng.integers(0, 2, (W, W), dtype=numpy.uint64)          # {0,1} rhs
dt_limb, r_limb = timed(lambda: matmul_limb(A, Bbin))
dt_single, r_single = timed(lambda: matmul_single(A, Bbin))
cross_ok = numpy.array_equal(r_single, r_limb)
print(f'  single float64 BLAS : {dt_single:7.3f}s  correct={cross_ok}  ({N*W*W/dt_single:.2e} ops/s)')
print(f'  limb-split 4xBLAS   : {dt_limb:7.3f}s  (reference)  ({N*W*W/dt_limb:.2e} ops/s)')

print()
print('op-count (BLAS N*W^2 vs four-russians precompute+gather, the subset-sum asymptotic):')
for t in [4, 8]:
    ops_c = (W // t) * N * (W + (1 << t))
    print(f'  t={t}: BLAS {N*W*W:>14,}  4Russ {ops_c:>14,}  -> {N*W*W/ops_c:5.1f}x fewer ops (O(N W^2/log W))')
print()
print('Reading: 4Russ does fewer ops (the log W / subset-sum asymptotic) but at')
print('gather/scatter throughput (~1e8/s) vs BLAS GEMM (~3e10/s), so single-float')
print('BLAS wins concretely.  The {0,1} structure pays both ways: fewer ops AND')
print('sums that fit in float64 -> BLAS.  Limb-split extends BLAS to full-range.')
