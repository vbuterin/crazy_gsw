"""Packed GSW -- k hidden eigenvectors in one ciphertext (SIMD packing).

Same skeleton as basic_gsw.py (k = 1); diff it to see exactly what packing
costs.  We keep k secret row-vectors s_0 .. s_{k-1} and ask the message matrix M
to be DIAGONAL with respect to them:

        S . M = diag(m) . S            (each s_i is a pseudo-eigenvector, eigenvalue m[i])

so one ciphertext carries a VECTOR of k messages, and ciphertext multiplication
is slot-wise (Hadamard) product.

THE ASSUMPTION, HONESTLY:  S = [solve | I_k] and Bbar = [B_solve; B_pin] with
        B_pin = -solve . B_solve + E          (E small)
i.e. each row of S is a NOISY NULL VECTOR of Bbar.  Stripping the public I_k
pin, the sub-matrix (B_solve, B_pin) is plain LWE with secret `solve` -- k
independent k-dim LWE instances sharing the W samples B_solve.  So the
EFFECTIVE secret dimension is k = N/2 (NOT N); the pins are public constants,
like the `1` in basic GSW's s = (-s0, 1).  The only difference from basic GSW
is "1 noisy null vector" vs "k noisy null vectors".  See README.

BRAVE PACKING:  dim = 0  =>  k = N/2  =>  N = 2k.  A ciphertext mul costs
N * W^2 = 8 k^3 qbits^2 mod-muls; the slot-wise product does k -> overhead
8 k^2 qbits^2.  (Matrix GSW cancels the extra k^2 by doing k^3 useful muls;
see matrix_gsw.py.)

KEYGEN HAS NO MATRIX INVERSE:  we sample Bbar uniformly from the affine coset
{B : S.B = E} by building a null-space basis of S from its entries (no solve^{-1},
no odd-determinant retry, no matinv_mod).  See keygen() and README.

Fast float64-BLAS arithmetic throughout (qbits <= 32).  See README.md.
"""
import numpy, time
from collections import namedtuple

# A scheme configuration.  Layout per secret row: [ free (dim) | solve (k) | pin (k) ].
# The brave default is dim = 0  =>  N = 2k  =>  k = N/2.
Params = namedtuple('Params', 'k dim qbits msgbits base_error')

def N(p):    return p.dim + 2 * p.k           # layout [ free (dim) | solve (k) | pin (k) ]
def W(p):    return N(p) * p.qbits            # gadget width = ciphertext cols
def MOD(p):  return 1 << p.qbits
def MM(p):   return 1 << p.msgbits

_rng = numpy.random.default_rng(0)
N_BLOCKS = 16                                # column-chunks for the optimized (low-memory) path

# ---- helpers (identical across all three files) ----
def random(shape, p):
    return _rng.integers(0, MOD(p), shape, dtype=numpy.uint64)

def random_low_norm(shape, p):
    return (_rng.integers(-p.base_error, p.base_error + 1, shape,
                          dtype=numpy.int64) % MOD(p)).astype(numpy.uint64)

def _random_low_norm_float(shape, p):
    # R as float64 {-1,0,1} -- generated in column-blocks to avoid a W x W int64
    # intermediate.  float64 entries pass through _to_float with zero copy.
    R = numpy.empty(shape, dtype=numpy.float64)
    blk = 8192
    for i in range(0, shape[1], blk):
        R[:, i:i+blk] = _rng.integers(-p.base_error, p.base_error + 1,
                                     (shape[0], min(blk, shape[1] - i))).astype(numpy.float64)
    return R

def zeros(shape):
    return numpy.zeros(shape, dtype=numpy.uint64)

def _to_float(x, p):
    # Convert to centered float64 for BLAS.  Float64 inputs pass through with
    # NO copy -- critical for the W x W matrices at large N.  uint64 inputs in
    # [0,q) go to float64 and are centered; values up to 2^32 are exact in f64.
    if x.dtype == numpy.float64:
        return x
    q = MOD(p)
    xf = numpy.float64(x)
    return numpy.where(xf > q // 2, xf - q, xf)

def matmul(A, B, p):
    # EVERY matmul goes through BLAS -- no slow uint64 path -- via two routes:
    #
    #  (1) SINGLE float64 GEMM when the product fits in the 53-bit mantissa:
    #        W * max|A| * max|B| < 2^53   (A,B centered to [-q/2, q/2)).
    #      Covers the {0,1}-rhs (G^{-1}), {0,+-1}-rhs (encryption pad R), and
    #      small-message cases -- one BLAS call, exact.
    #
    #  (2) LIMB-SPLIT into 16-bit halves otherwise (full-range @ full-range:
    #      decrypt S.ct):
    #        (A_lo + 2^16 A_hi)(B_lo + 2^16 B_hi) = LL + 2^16(LH+HL) + 2^32 HH
    #      Each sub-product < 2^16 * 2^16 * W = 2^46 < 2^53 -> exact in float64,
    #      and 2^32 HH vanishes mod 2^qbits (qbits <= 32).  Four BLAS calls.
    As = _to_float(A, p)
    w = A.shape[1]
    amax = max(float(As.max()), float(-As.min())) if As.size else 0.0
    if B.dtype == numpy.float64:
        bmax = max(float(B.max()), float(-B.min())) if B.size else 0.0
    else:
        bmax_raw = int(B.max()) if B.size else 0
        bmax = max(bmax_raw, MOD(p) - bmax_raw)
    if amax * bmax * w < (1 << 53):
        return numpy.uint64((As @ _to_float(B, p)) % MOD(p))
    if A.dtype != numpy.uint64: A = numpy.uint64(A) % MOD(p)
    if B.dtype != numpy.uint64: B = numpy.uint64(B) % MOD(p)
    split = 16; mask = (1 << split) - 1
    A_lo = numpy.float64(A & mask); A_hi = numpy.float64(A >> split)
    B_lo = numpy.float64(B & mask); B_hi = numpy.float64(B >> split)
    LL = numpy.uint64(A_lo @ B_lo)
    cross = (numpy.uint64(A_lo @ B_hi) + numpy.uint64(A_hi @ B_lo)) << split
    return (LL + cross) % MOD(p)

def center(x, p):
    q = MOD(p)
    return ((x.astype(numpy.int64) + q // 2) % q - q // 2)

# ---- gadget ----
def gadget(n, qbits):
    x = numpy.zeros((n, n * qbits), dtype=numpy.uint64)
    for i in range(n):
        for j in range(qbits):
            x[i, i * qbits + j] = 1 << j
    return x

def G(p):
    return gadget(N(p), p.qbits)

def gadget_inv(inp, qbits):
    # Full bit-decomposition -> float64 {0,1}, shape (n*qbits, W).  Used by the
    # BASIC mul_cts.  The OPTIMIZED mul_cts uses _gadget_inv_cols instead, which
    # only materializes one column-chunk at a time (memory: W x W/N_BLOCKS).
    o = numpy.zeros((inp.shape[0] * qbits, inp.shape[1]), dtype=numpy.float64)
    for i in range(qbits):
        o[i::qbits, :] = numpy.float64((inp >> i) & 1)
    return o

def _gadget_inv_cols(inp, qbits, cols):
    # Bit-decompose only a column-slice of inp -> float64 {0,1}.  Same mapping
    # as gadget_inv (rows i::qbits = bit i) but on inp[:, cols], so the columns
    # of G^{-1} line up with the columns of the output ciphertext.  This is what
    # makes column-blocking of mul_cts exact: G^{-1}(C2)[:, cols] == this.
    c = inp[:, cols]
    o = numpy.zeros((c.shape[0] * qbits, c.shape[1]), dtype=numpy.float64)
    for i in range(qbits):
        o[i::qbits, :] = numpy.float64((c >> i) & 1)
    return o

# ---- keygen (null-space / coset sampling -- NO matrix inverse) ----
# S = [ free (random) | solve (random, ANY) | pin (I_k) ]   (k x N)
# We need S . Bbar = E.  {B : S.B = E} is an affine coset of the right null
# space of S.  A basis N for that null space is built BY CONSTRUCTION from S's
# entries (no inverse): each column of N satisfies S . (column) = 0.
#
#   N = [ I_dim    0    ]  free rows        (dim of these)
#       [ 0       I_k   ]  solve rows       (k of these)
#       [ -free  -solve ]  pin rows         (k of these)
#
# so S . N = 0.  Writing X = [X_free; X_solve] random,  Bbar = N . X + [0;0;E]:
#     B_free  = X_free                       (random)
#     B_solve = X_solve                      (random)
#     B_pin   = E - free.X_free - solve.X_solve
# and S . Bbar = E.  NO inverse of solve -- works for ANY solve (singular or
# not), and Bbar is UNIFORM over the coset {B : S.B = E}, the same distribution
# the old solve^{-1} scheme produced.  No matinv_mod, no retry loop, no odd det.
def keygen(p):
    free  = random((p.k, p.dim), p)
    solve = random((p.k, p.k), p)                                           # any random matrix
    pin   = numpy.eye(p.k, dtype=numpy.uint64)
    S = numpy.concatenate([free, solve, pin], axis=1)                       # k x N
    X_free  = random((p.dim, W(p)), p)                                      # -> B_free
    X_solve = random((p.k,  W(p)), p)                                       # -> B_solve
    E = random_low_norm((p.k, W(p)), p)
    B_free  = X_free
    B_solve = X_solve
    B_pin   = (E - matmul(free,  X_free,  p)                                # E - free.X_free - solve.X_solve
                    - matmul(solve, X_solve, p)) % MOD(p)
    PUBKEY = numpy.concatenate([B_free, B_solve, B_pin], axis=0)            # N x W
    return S, PUBKEY

# ---- encrypt (basic vs optimized) ----
# m = vector of k messages (a DIAGONAL message matrix).  Build M (N x N) with
# S . M = diag(m) . S via the null-space trick (no inverse): pick M_free,
# M_solve random, set M_pin = diag(m).S - free.M_free - solve.M_solve.
# Encryption needs the secret (M depends on S) -- unlike basic GSW.
def _build_M(messages, p, S):
    m = numpy.array(messages, dtype=numpy.uint64) % MM(p)                   # length k
    M_free  = random((p.dim, N(p)), p)
    M_solve = random((p.k,   N(p)), p)
    M_pin   = (numpy.diag(m) @ S                                            # diag(m).S
               - matmul(S[:, :p.dim],           M_free,  p)               # - free.M_free
               - matmul(S[:, p.dim:p.dim + p.k], M_solve, p)) % MOD(p)     # - solve.M_solve
    return numpy.concatenate([M_free, M_solve, M_pin], axis=0)             # N x N

def encrypt_basic(messages, p, S, PUBKEY):
    # Clear version: materialize the full W x W encryption pad R, then one GEMM.
    # Peak memory ~ one W x W array (34 GB at N=2048) -- fine for toy, tight at
    # full width.
    M = _build_M(messages, p, S)
    R = _random_low_norm_float((W(p), W(p)), p)                             # W x W  <-- the hog
    return (matmul(PUBKEY, R, p) + matmul(M, G(p), p)) % MOD(p)             # N x W

def encrypt_opt(messages, p, S, PUBKEY):
    # Memory-optimized: process R in N_BLOCKS column-chunks.  Each chunk is
    # W x (W/N_BLOCKS); PUBKEY @ R_chunk accumulates into the matching output
    # columns.  M . G is small (N x W) so it is done once.  Peak memory
    # ~ W x W/N_BLOCKS + a few N x W arrays (~2 GB at N=2048 instead of 34 GB),
    # and the result is bit-identical to encrypt_basic (column-blocking is exact).
    M = _build_M(messages, p, S)
    out = matmul(M, G(p), p)                                                # N x W, the message part
    nblk = N_BLOCKS; wc = W(p) // nblk
    for b in range(nblk):
        cols = slice(b * wc, (b + 1) * wc)
        R_chunk = _random_low_norm_float((W(p), wc), p)                     # W x (W/nblk)
        out[:, cols] = (out[:, cols] + matmul(PUBKEY, R_chunk, p)) % MOD(p)
    return out

# ---- decrypt ----
# S . C ~ diag(m) . (S . G).  Row i, at pin block p_i, equals m[i] . (1,2,4,...).
def decrypt(ct, p, S):
    SC = center(matmul(S, ct, p), p)                                        # k x W
    jj = p.qbits - p.msgbits - 1
    out = []
    for i in range(p.k):
        pin = p.dim + p.k + i                                               # s_i's pinned coordinate
        v = int(SC[i, pin * p.qbits + jj])
        v = ((v + MOD(p) // 2) % MOD(p)) - MOD(p) // 2
        out.append(int(round(v / (1 << jj))) % MM(p))
    return out

# ---- homomorphic ops (basic vs optimized) ----
def add_cts(ct1, ct2, p):
    return (ct1 + ct2) % MOD(p)

def mul_cts_basic(ct1, ct2, p):
    # Clear version: materialize the full W x W bit-decomposition G^{-1}(ct2),
    # then one GEMM.  Peak memory ~ one W x W array.
    return matmul(ct1, gadget_inv(ct2, p.qbits), p)

def mul_cts_opt(ct1, ct2, p):
    # Memory-optimized: block the W x W G^{-1}(ct2) into N_BLOCKS column-chunks.
    # Bit-decompose and multiply one chunk at a time.  Exact (column-blocking).
    nblk = N_BLOCKS; wc = W(p) // nblk
    out = numpy.zeros((N(p), W(p)), dtype=numpy.uint64)
    for b in range(nblk):
        cols = slice(b * wc, (b + 1) * wc)
        Ginv_chunk = _gadget_inv_cols(ct2, p.qbits, cols)                   # W x (W/nblk), float64 {0,1}
        out[:, cols] = matmul(ct1, Ginv_chunk, p)
    return out

# ---- timing helper ----
def _avg(fn, reps):
    t0 = time.perf_counter()
    r = None
    for _ in range(reps):
        r = fn()
    return (time.perf_counter() - t0) / reps, r

# ---- run one config: correctness + concrete runtime comparison ----
def _run(p, label, m1, m2, both_methods=True):
    qbits, msgbits, k, dim = p.qbits, p.msgbits, p.k, p.dim
    print(f'\n=== {label}:  k={k}  dim={dim}  N={N(p)}  W={W(p)}  qbits={qbits}  msgbits={msgbits} ===')
    t0 = time.perf_counter(); S, PUBKEY = keygen(p); dt_key = time.perf_counter() - t0
    noise = int(numpy.abs(center(matmul(S, PUBKEY, p), p)).max())
    print(f'  keygen      {dt_key:8.3f}s   noise floor |S.Bbar| = {noise}')

    enc = [encrypt_opt, encrypt_basic] if both_methods else [encrypt_opt]
    mul = [mul_cts_opt,   mul_cts_basic] if both_methods else [mul_cts_opt]
    for ei, efn in enumerate(enc):
        tag = efn.__name__.replace('encrypt_', '')
        dt_e1, ct1 = _avg(lambda f=efn: f(m1, p, S, PUBKEY), 1)
        dt_e2, ct2 = _avg(lambda f=efn: f(m2, p, S, PUBKEY), 1)
        ok_e = (decrypt(ct1, p, S) == list(numpy.array(m1) % MM(p)))
        print(f'  encrypt[{tag}] x2  {dt_e1 + dt_e2:8.3f}s   decrypt m1 ok={ok_e}')
        for mi, mfn in enumerate(mul):
            mtag = mfn.__name__.replace('mul_cts_', '')
            dt_add, ctA = _avg(lambda: add_cts(ct1, ct2, p), 1)
            dt_mul, ctM = _avg(lambda mfn=mfn: mfn(ct1, ct2, p), 1)
            dt_decA, dA = _avg(lambda: decrypt(ctA, p, S), 1)
            dt_decM, dM = _avg(lambda: decrypt(ctM, p, S), 1)
            ok_a = (dA == [(a + b) % MM(p) for a, b in zip(m1, m2)])
            ok_m = (dM == [(a * b) % MM(p) for a, b in zip(m1, m2)])
            print(f'  mul[{mtag}]   ct_add {dt_add:7.3f}s  ct_mul {dt_mul:7.3f}s  '
                  f'decrypt add ok={ok_a}  mul ok={ok_m}')

    # ---- plaintext op (slot-wise add/mul of two k-vectors mod 2^msgbits) ----
    a = numpy.array(m1, dtype=numpy.uint64); b = numpy.array(m2, dtype=numpy.uint64)
    mm = numpy.uint64(MM(p))
    reps = 100_000
    dt_pa, _ = _avg(lambda: (a + b) % mm, reps)
    dt_pm, _ = _avg(lambda: (a * b) % mm, reps)
    dt_mul_opt, _ = _avg(lambda: mul_cts_opt(ct1, ct2, p), 1)
    print(f'  pt_add      {dt_pa * 1e6:8.2f}us   pt_mul {dt_pm * 1e6:8.2f}us   (k={k} slot ops, {reps} reps)')
    print(f'  OVERHEAD    add: {dt_add / dt_pa:>10,.0f}x    mul: {dt_mul_opt / dt_pm:>10,.0f}x   (wall-clock, optimized)')
    # ---- theoretical overheads (three honest notions; see README) ----
    ct_ops  = N(p) * W(p) * W(p)             # one ct mul = N * W^2 mod-muls mod 2^qbits
    pt_ops  = k                              # one slot-wise product = k mod-muls
    msg_bits = k * msgbits                   # k message slots
    op_count = ct_ops // pt_ops             # 8 k^2 qbits^2 with k=N/2 (qbits^2 = gadget expansion)
    bit_op  = op_count * (qbits * qbits) // (msgbits * msgbits)   # schoolbook bit-ops
    storage = (N(p) * W(p) * qbits) // msg_bits                  # ct bits per message bit
    print(f'  THEORY  op-count = {op_count:>14,}x   bit-op = {bit_op:>14,}x   storage = {storage:>10,}x')

def test():
    print('packed GSW -- k hidden eigenvectors, slot-wise (Hadamard) ops')
    _run(Params(k=3,  dim=0, qbits=32, msgbits=4, base_error=1), 'TOY',
         m1=[1, 2, 3], m2=[2, 1, 0], both_methods=True)
    _run(Params(k=1024, dim=0, qbits=32, msgbits=8, base_error=1), 'FULL-ON',
         m1=list(range(1024)), m2=list(range(1024)), both_methods=False)   # opt only (memory)
    print()
    print('full-on: N=2048=2k, effective LWE dim k=1024 at qbits=32 (~100-128-bit conjectured;')
    print('see README: the I_k pins are public, so the assumption is k independent k-dim LWE')
    print('instances, not one 2k-dim instance).  Packed overhead 8 k^2 qbits^2 grows with k.')

if __name__ == '__main__':
    test()
