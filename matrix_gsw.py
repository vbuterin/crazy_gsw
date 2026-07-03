"""Matrix GSW -- a hidden invariant subspace in one ciphertext.

Same skeleton as packed_gsw.py; diff it to see the whole generalization.
Packed GSW forces the message matrix M to be DIAGONAL w.r.t. the secret rows
(each s_i is a pseudo-eigenvector).  Matrix GSW drops that restriction: the
message is a full k x k matrix m and we only require

        S . M = m . S            (rows of S are NOT eigenvectors; their span
                                  is an m-invariant subspace)

so  S . C ~ m . (S . G).  Decryption reads k^2 values (one per (row, pin) pair),
and ciphertext multiplication composes by ordinary MATRIX product:

        S . (C1 . G^{-1}(C2)) = m1 . S . G . G^{-1}(C2) = m1 . S . C2 ~ (m1 @ m2) . S . G

so mul_cts encodes m1 @ m2 (non-commutative).  Diagonal m recovers packed GSW;
k = 1 recovers basic GSW.  The keygen and public key are IDENTICAL to packed
GSW, so matrix GSW adds NO new structural leakage over packed GSW.

THE ASSUMPTION, HONESTLY (same as packed GSW):  S = [solve | I_k], Bbar =
[B_solve; B_pin] with B_pin = -solve.B_solve + E.  Stripping the public I_k pin,
this is k independent k-dim LWE instances sharing the W samples B_solve.  The
EFFECTIVE secret dimension is k = N/2, NOT N -- the pins are public constants
like the `1` in basic GSW's s = (-s0, 1).  See README.

BRAVE PACKING & THE CLEAN OVERHEAD:  with dim = 0, k = N/2, N = 2k.  A ciphertext
mul costs N * W^2 = 8 k^3 qbits^2 mod-muls; a plaintext k x k matmul costs k^3.
The k^3 CANCELS, giving an op-count overhead

        overhead  =  8 * qbits^2           (mod-mul count, INDEPENDENT of k and N)

i.e. packing is asymptotically FREE -- you carry a k x k matrix at the same
per-mul op-count overhead as a single scalar.  The qbits^2 is the gadget
expansion (W = N*qbits -> W^2).  The plaintext modulus does NOT divide this time
overhead (a small-plaintext mod-mul is still one instruction); msgbits enters
only STORAGE (4 qbits^2 / msgbits) and schoolbook bit-ops (8 qbits^4 / msgbits^2).
See README for the full three-notions discussion.

KEYGEN HAS NO MATRIX INVERSE:  coset sampling via a null-space basis of S -- no
solve^{-1}, no odd-determinant retry, no matinv_mod.  See keygen() and README.

Fast float64-BLAS arithmetic throughout (qbits <= 32).  See README.md.
"""
import numpy, time
from collections import namedtuple

# A scheme configuration.  Layout per secret row: [ free (dim) | solve (k) | pin (k) ].
# The brave default is dim = 0  =>  N = 2k  =>  k = N/2.  Identical to packed GSW.
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
    # BASIC mul_cts.  The OPTIMIZED mul_cts uses _gadget_inv_cols instead.
    o = numpy.zeros((inp.shape[0] * qbits, inp.shape[1]), dtype=numpy.float64)
    for i in range(qbits):
        o[i::qbits, :] = numpy.float64((inp >> i) & 1)
    return o

def _gadget_inv_cols(inp, qbits, cols):
    # Bit-decompose only a column-slice of inp -> float64 {0,1}.  Same mapping
    # as gadget_inv (rows i::qbits = bit i) but on inp[:, cols], so columns of
    # G^{-1} line up with the output ciphertext columns.  Makes column-blocking
    # of mul_cts exact: G^{-1}(C2)[:, cols] == this.
    c = inp[:, cols]
    o = numpy.zeros((c.shape[0] * qbits, c.shape[1]), dtype=numpy.float64)
    for i in range(qbits):
        o[i::qbits, :] = numpy.float64((c >> i) & 1)
    return o

# ---- keygen (null-space / coset sampling -- NO matrix inverse) ----
# IDENTICAL to packed GSW keygen.  S = [ free | solve | I_k ], and Bbar is
# sampled uniformly from {B : S.B = E} via a null-space basis of S (no inverse):
#     B_free  = X_free                       (random)
#     B_solve = X_solve                      (random)
#     B_pin   = E - free.X_free - solve.X_solve
# Works for ANY solve (singular or not); same LWE-coset distribution as the old
# solve^{-1} scheme.  No matinv_mod, no retry loop, no odd det.
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
# m = k x k message MATRIX (no diagonal restriction).  Build M (N x N) with
# S . M = m . S via the null-space trick (no inverse): pick M_free, M_solve
# random, set M_pin = m.S - free.M_free - solve.M_solve.  The ONLY change vs
# packed GSW is `numpy.diag(m) @ S` -> `m @ S`.  Encryption needs the secret.
def _build_M(m, p, S):
    m = numpy.array(m, dtype=numpy.uint64) % MM(p)                          # k x k
    M_free  = random((p.dim, N(p)), p)
    M_solve = random((p.k,   N(p)), p)
    M_pin   = (m @ S                                                         # m . S  (not diag(m) . S)
               - matmul(S[:, :p.dim],           M_free,  p)                # - free.M_free
               - matmul(S[:, p.dim:p.dim + p.k], M_solve, p)) % MOD(p)      # - solve.M_solve
    return numpy.concatenate([M_free, M_solve, M_pin], axis=0)              # N x N

def encrypt_basic(m, p, S, PUBKEY):
    # Clear version: materialize the full W x W encryption pad R, then one GEMM.
    # Peak memory ~ one W x W array (34 GB at N=2048).
    M = _build_M(m, p, S)
    R = _random_low_norm_float((W(p), W(p)), p)                             # W x W  <-- the hog
    return (matmul(PUBKEY, R, p) + matmul(M, G(p), p)) % MOD(p)             # N x W

def encrypt_opt(m, p, S, PUBKEY):
    # Memory-optimized: process R in N_BLOCKS column-chunks.  Each chunk is
    # W x (W/N_BLOCKS); PUBKEY @ R_chunk accumulates into the matching output
    # columns.  M . G is small (N x W), done once.  Peak ~ W x W/N_BLOCKS + a
    # few N x W arrays (~2 GB at N=2048 instead of 34 GB).  Bit-identical to
    # encrypt_basic (column-blocking is exact).
    M = _build_M(m, p, S)
    out = matmul(M, G(p), p)                                                # N x W, the message part
    nblk = N_BLOCKS; wc = W(p) // nblk
    for b in range(nblk):
        cols = slice(b * wc, (b + 1) * wc)
        R_chunk = _random_low_norm_float((W(p), wc), p)                     # W x (W/nblk)
        out[:, cols] = (out[:, cols] + matmul(PUBKEY, R_chunk, p)) % MOD(p)
    return out

# ---- decrypt ----
# S . C ~ m . (S . G).  Row i, at pin block p_l, equals m[i,l] . (1,2,4,...):
# (s_j . G) at pin block p_l is delta_{jl} . g, so the sum over j picks m[i,l].
def decrypt(ct, p, S):
    SC = center(matmul(S, ct, p), p)                                        # k x W
    jj = p.qbits - p.msgbits - 1
    out = numpy.zeros((p.k, p.k), dtype=numpy.int64)
    for i in range(p.k):
        for l in range(p.k):
            pin = p.dim + p.k + l                                           # s_l's pinned coordinate
            v = int(SC[i, pin * p.qbits + jj])
            v = ((v + MOD(p) // 2) % MOD(p)) - MOD(p) // 2
            out[i, l] = int(round(v / (1 << jj))) % MM(p)
    return out

# ---- homomorphic ops (basic vs optimized) ----
def add_cts(ct1, ct2, p):
    return (ct1 + ct2) % MOD(p)

def mul_cts_basic(ct1, ct2, p):
    # Clear version: materialize the full W x W bit-decomposition G^{-1}(ct2).
    # Peak memory ~ one W x W array.
    return matmul(ct1, gadget_inv(ct2, p.qbits), p)

def mul_cts_opt(ct1, ct2, p):
    # Memory-optimized: block G^{-1}(ct2) into N_BLOCKS column-chunks.  Bit-
    # decompose and multiply one chunk at a time.  Exact (column-blocking).
    nblk = N_BLOCKS; wc = W(p) // nblk
    out = numpy.zeros((N(p), W(p)), dtype=numpy.uint64)
    for b in range(nblk):
        cols = slice(b * wc, (b + 1) * wc)
        Ginv_chunk = _gadget_inv_cols(ct2, p.qbits, cols)                   # W x (W/nblk), float64 {0,1}
        out[:, cols] = matmul(ct1, Ginv_chunk, p)
    return out

# ---- convenience constructors (special cases of the general matrix message) ----
def encrypt_vector(vec, p, S, PUBKEY, opt=True):   # diagonal message == packed GSW
    f = encrypt_opt if opt else encrypt_basic
    return f(numpy.diag(vec), p, S, PUBKEY)

def encrypt_scalar(a, p, S, PUBKEY, opt=True):     # a . I  ->  acts as scalar multiplication
    f = encrypt_opt if opt else encrypt_basic
    return f(int(a) * numpy.eye(p.k, dtype=numpy.uint64), p, S, PUBKEY)

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

    M1 = numpy.array(m1, dtype=numpy.uint64) % MM(p); M2 = numpy.array(m2, dtype=numpy.uint64) % MM(p)
    enc = [encrypt_opt, encrypt_basic] if both_methods else [encrypt_opt]
    mul = [mul_cts_opt,   mul_cts_basic] if both_methods else [mul_cts_opt]
    for efn in enc:
        tag = efn.__name__.replace('encrypt_', '')
        dt_e1, ct1 = _avg(lambda f=efn: f(m1, p, S, PUBKEY), 1)
        dt_e2, ct2 = _avg(lambda f=efn: f(m2, p, S, PUBKEY), 1)
        ok_e = numpy.array_equal(decrypt(ct1, p, S), M1)
        print(f'  encrypt[{tag}] x2  {dt_e1 + dt_e2:8.3f}s   decrypt m1 ok={ok_e}')
        for mfn in mul:
            mtag = mfn.__name__.replace('mul_cts_', '')
            dt_add, ctA = _avg(lambda: add_cts(ct1, ct2, p), 1)
            dt_mul, ctM = _avg(lambda mfn=mfn: mfn(ct1, ct2, p), 1)
            dt_decA, dA = _avg(lambda: decrypt(ctA, p, S), 1)
            dt_decM, dM = _avg(lambda: decrypt(ctM, p, S), 1)
            ok_a = numpy.array_equal(dA, (M1 + M2) % MM(p))
            ok_m = numpy.array_equal(dM, (M1 @ M2) % MM(p))
            print(f'  mul[{mtag}]   ct_add {dt_add:7.3f}s  ct_mul {dt_mul:7.3f}s  '
                  f'decrypt add ok={ok_a}  mul ok={ok_m}')

    # ---- plaintext op (k x k matmul / add mod 2^msgbits) -- ALSO BLAS (small msg fits f64) ----
    a = M1; b = M2; mm = numpy.uint64(MM(p))
    reps = 100
    dt_pa, _ = _avg(lambda: (a + b) % mm, reps)
    dt_pm, _ = _avg(lambda: numpy.uint64(numpy.float64(a) @ numpy.float64(b)) % mm, reps)
    dt_mul_opt, _ = _avg(lambda: mul_cts_opt(ct1, ct2, p), 1)
    print(f'  pt_add      {dt_pa * 1e3:8.3f}ms   pt_mul {dt_pm * 1e3:8.3f}ms   (k={k} BLAS matmul, {reps} reps)')
    print(f'  OVERHEAD    add: {dt_add / dt_pa:>10,.0f}x    mul: {dt_mul_opt / dt_pm:>10,.0f}x   (wall-clock, optimized)')
    # ---- theoretical overheads (three honest notions; see README) ----
    ct_ops  = N(p) * W(p) * W(p)             # one ct mul = N * W^2 mod-muls
    pt_ops  = k * k * k                      # one k x k matmul = k^3 mod-muls
    msg_bits = k * k * msgbits               # k x k message entries
    op_count = ct_ops // pt_ops             # = 8 qbits^2 with k=N/2 (k^3 cancels!)
    bit_op  = op_count * (qbits * qbits) // (msgbits * msgbits)   # schoolbook bit-ops
    storage = (N(p) * W(p) * qbits) // msg_bits                  # ct bits per message bit
    print(f'  THEORY  op-count = {op_count:>14,}x   bit-op = {bit_op:>14,}x   storage = {storage:>10,}x')

def test():
    print('matrix GSW -- hidden invariant subspace, matmul composition')
    # toy: rich correctness on small non-commuting matrices
    _run(Params(k=3, dim=0, qbits=32, msgbits=4, base_error=1), 'TOY',
         m1=[[1, 2, 0], [0, 1, 3], [2, 0, 1]],
         m2=[[1, 0, 1], [1, 1, 0], [0, 1, 1]], both_methods=True)
    # full-on: k=1024 (N=2048), binary matrices so the k x k product fits in
    # msgbits=8 mod 256, and binary ||m||_inf=1 keeps mul-noise ~2^15.2 << 2^23 window.
    rng = numpy.random.default_rng(42)
    k = 1024
    M1 = rng.integers(0, 2, (k, k), dtype=numpy.uint64)
    M2 = rng.integers(0, 2, (k, k), dtype=numpy.uint64)
    _run(Params(k=k, dim=0, qbits=32, msgbits=8, base_error=1), 'FULL-ON',
         m1=M1.tolist(), m2=M2.tolist(), both_methods=False)                 # opt only (memory)
    print()
    print('full-on: N=2048=2k, effective LWE dim k=1024 at qbits=32 (~100-128-bit conjectured;')
    print('the I_k pins are public, so the assumption is k independent k-dim LWE instances,')
    print('not one 2k-dim instance).  k=N/2 makes k^3 cancel, so the op-count overhead')
    print('8 qbits^2 = 8192x is N-INDEPENDENT -- only wall-clock scales as N^3.')

if __name__ == '__main__':
    test()
