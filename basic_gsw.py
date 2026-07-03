"""GSW with a single hidden eigenvector -- the k = 1 base case (clean LWE).

A GSW ciphertext is a matrix C whose secret key s is an approximate left
eigenvector and whose eigenvalue is the message m:

        s . C  ~  m . (s . G)

We build  C = Bbar . R + M . G  with  M = m . I_N  (a scalar multiple of the
identity), which is SECRET-INDEPENDENT, so encryption needs only the public
key.  The public key is built the clean LWE way: the pinned row is
s0 . B_free + e, so  s . Bbar = (-s0, 1) . Bbar = e ~ 0.  This is the one
clean-LWE instance in the family; compare keygen here to keygen in
packed_gsw.py / matrix_gsw.py to see the price of packing more than one message.

This file uses fast uint64 numpy arithmetic throughout (qbits <= 32 so that
q^2 fits in 64 bits and intermediate matmul overflow mod 2^64 is harmless once
we reduce mod 2^qbits, since 2^qbits | 2^64).  See README.md for the full story.
"""
import numpy, time
from collections import namedtuple

# A scheme configuration.  basic GSW always has k = 1; `dim` is the hidden
# (clean-LWE) part of the secret and N = dim + 1.
Params = namedtuple('Params', 'k dim qbits msgbits base_error')

def N(p):    return p.dim + 1                # layout [ free (dim) | pin (1) ]
def W(p):    return N(p) * p.qbits           # gadget width = ciphertext cols
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
    # intermediate (34GB at N=2048).  float64 entries pass through _to_float with
    # zero copy, so the matmul never materializes a separate float64 copy of R.
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
    # NO copy -- critical for the W x W matrices at large N, where a duplicate
    # 34GB copy of the encryption randomness R would blow memory.  uint64 inputs
    # in [0,q) go directly to float64 and are centered; values up to 2^32 are
    # exact in float64 (2^32 < 2^53).
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
    #      small-message cases -- one BLAS call, exact.  R and G^{-1} are stored
    #      as float64 (entries {-1,0,1} / {0,1}) so _to_float passes them through
    #      with zero copy, keeping peak memory at ~one W x W array.
    #
    #  (2) LIMB-SPLIT into 16-bit halves otherwise (full-range @ full-range:
    #      decrypt S.ct):
    #        (A_lo + 2^16 A_hi)(B_lo + 2^16 B_hi) = LL + 2^16(LH+HL) + 2^32 HH
    #      Each sub-product < 2^16 * 2^16 * W = 2^46 < 2^53 -> exact in float64,
    #      and 2^32 HH vanishes mod 2^qbits (qbits <= 32).  Four BLAS calls, no
    #      multi-precision, no uint64 matmul.  Valid for W < 2^21 (crypto sizes).
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
    # as gadget_inv (rows i::qbits = bit i) but on inp[:, cols].  Makes column-
    # blocking of mul_cts exact: G^{-1}(C2)[:, cols] == this.
    c = inp[:, cols]
    o = numpy.zeros((c.shape[0] * qbits, c.shape[1]), dtype=numpy.float64)
    for i in range(qbits):
        o[i::qbits, :] = numpy.float64((c >> i) & 1)
    return o

# ---- keygen (clean LWE) ----
# s = (-s0, 1).  Bbar = [ B_free ]   (dim x W)  -- random
#                     [ B_pin  ]   (1   x W)  -- B_pin = s0 . B_free + e
# so  s . Bbar = -s0 . B_free + B_pin = e ~ 0.  Under LWE, Bbar ~ uniform.
def keygen(p):
    s0 = random((p.dim,), p)
    B_free = random((p.dim, W(p)), p)
    e = random_low_norm((W(p),), p)
    B_pin = (s0 @ B_free + e) % MOD(p)
    PUBKEY = numpy.concatenate([B_free, B_pin.reshape(1, -1)], axis=0)      # (N, W)
    secret = numpy.concatenate([(-s0) % MOD(p), numpy.ones(1, dtype=numpy.uint64)]).reshape(1, -1)
    return secret, PUBKEY

# ---- encrypt (basic vs optimized) ----
# m (scalar).  M = m . I_N is secret-independent -> only the public key needed.
def encrypt_basic(message, p, PUBKEY):
    # Clear version: materialize the full W x W encryption pad R, then one GEMM.
    # Peak memory ~ one W x W array (34 GB at N=2048).
    M = (int(message) * numpy.eye(N(p), dtype=numpy.uint64)) % MOD(p)
    R = _random_low_norm_float((W(p), W(p)), p)                             # W x W  <-- the hog
    return (matmul(PUBKEY, R, p) + matmul(M, G(p), p)) % MOD(p)             # N x W

def encrypt_opt(message, p, PUBKEY):
    # Memory-optimized: process R in N_BLOCKS column-chunks.  Each chunk is
    # W x (W/N_BLOCKS); PUBKEY @ R_chunk accumulates into the matching output
    # columns.  M . G is small (N x W), done once.  Peak ~ W x W/N_BLOCKS + a
    # few N x W arrays (~2 GB at N=2048 instead of 34 GB).  Bit-identical to
    # encrypt_basic (column-blocking is exact).
    M = (int(message) * numpy.eye(N(p), dtype=numpy.uint64)) % MOD(p)
    out = matmul(M, G(p), p)                                                # N x W, the message part
    nblk = N_BLOCKS; wc = W(p) // nblk
    for b in range(nblk):
        cols = slice(b * wc, (b + 1) * wc)
        R_chunk = _random_low_norm_float((W(p), wc), p)                     # W x (W/nblk)
        out[:, cols] = (out[:, cols] + matmul(PUBKEY, R_chunk, p)) % MOD(p)
    return out

# ---- decrypt ----
# s . C ~ m . (s . G).  At the pin block, s . G = (1, 2, 4, ...), a KNOWN
# constant, so that block equals m . (1, 2, 4, ...).  Read a low gadget
# coordinate for headroom, divide, reduce mod the message ring.
def decrypt(ct, p, secret):
    SC = center(matmul(secret, ct, p), p)                                   # (1, W)
    jj = p.qbits - p.msgbits - 1
    pin = p.dim                                                             # the single pin coordinate
    v = int(SC[0, pin * p.qbits + jj])
    v = ((v + MOD(p) // 2) % MOD(p)) - MOD(p) // 2
    return int(round(v / (1 << jj))) % MM(p)

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
    t0 = time.perf_counter(); secret, PUBKEY = keygen(p); dt_key = time.perf_counter() - t0
    noise = int(numpy.abs(center(matmul(secret, PUBKEY, p), p)).max())
    print(f'  keygen      {dt_key:8.3f}s   noise floor |s.Bbar| = {noise}')

    enc = [encrypt_opt, encrypt_basic] if both_methods else [encrypt_opt]
    mul = [mul_cts_opt,   mul_cts_basic] if both_methods else [mul_cts_opt]
    for efn in enc:
        tag = efn.__name__.replace('encrypt_', '')
        dt_e1, ct1 = _avg(lambda f=efn: f(m1, p, PUBKEY), 1)
        dt_e2, ct2 = _avg(lambda f=efn: f(m2, p, PUBKEY), 1)
        ok_e = (decrypt(ct1, p, secret) == m1 % MM(p))
        print(f'  encrypt[{tag}] x2  {dt_e1 + dt_e2:8.3f}s   ({dt_e1:.3f}+{dt_e2:.3f})  decrypt m1 ok={ok_e}')
        for mfn in mul:
            mtag = mfn.__name__.replace('mul_cts_', '')
            dt_add, ctA = _avg(lambda: add_cts(ct1, ct2, p), 1)
            dt_mul, ctM = _avg(lambda mfn=mfn: mfn(ct1, ct2, p), 1)
            dt_decA, dA = _avg(lambda: decrypt(ctA, p, secret), 1)
            dt_decM, dM = _avg(lambda: decrypt(ctM, p, secret), 1)
            ok_a = (dA == (m1 + m2) % MM(p)); ok_m = (dM == (m1 * m2) % MM(p))
            print(f'  mul[{mtag}]   ct_add {dt_add:7.3f}s  ct_mul {dt_mul:7.3f}s  '
                  f'decrypt add={dA} (ok={ok_a})  mul={dM} (ok={ok_m})')

    # ---- plaintext op (a single scalar mul/add mod 2^msgbits) ----
    a = numpy.uint64(m1); b = numpy.uint64(m2); mm = numpy.uint64(MM(p))
    reps = 1_000_000
    dt_pa, _ = _avg(lambda: (a + b) % mm, reps)
    dt_pm, _ = _avg(lambda: (a * b) % mm, reps)
    dt_mul_opt, _ = _avg(lambda: mul_cts_opt(ct1, ct2, p), 1)
    print(f'  pt_add      {dt_pa * 1e9:8.1f}ns   pt_mul {dt_pm * 1e9:8.1f}ns   (1 scalar op, {reps} reps)')
    print(f'  OVERHEAD    add: {dt_add / dt_pa:>10,.0f}x    mul: {dt_mul_opt / dt_pm:>10,.0f}x   (wall-clock, optimized)')
    # ---- theoretical overheads (three honest notions; see README) ----
    qbits, msgbits = p.qbits, p.msgbits
    ct_ops  = N(p) * W(p) * W(p)             # one ct mul = N * W^2 mod-muls mod 2^qbits
    pt_ops  = 1                             # one scalar mul
    msg_bits = msgbits                       # one scalar message
    op_count = ct_ops // pt_ops              # 1 mod-mul = 1 unit (fair CPU metric)
    bit_op  = op_count * (qbits * qbits) // (msgbits * msgbits)   # schoolbook bit-ops
    storage = (N(p) * W(p) * qbits) // msg_bits                  # ct bits per message bit
    print(f'  THEORY  op-count = {op_count:>14,}x   bit-op = {bit_op:>14,}x   storage = {storage:>10,}x')

def test():
    print('basic GSW -- k = 1, clean LWE')
    _run(Params(k=1, dim=2,   qbits=32, msgbits=4, base_error=1), 'TOY',     m1=11,  m2=7,  both_methods=True)
    _run(Params(k=1, dim=2047, qbits=32, msgbits=8, base_error=1), 'FULL-ON', m1=123, m2=200, both_methods=False)
    print()
    print('full-on: N=2048, clean LWE with n = dim = 2047 secret dimensions (~128-bit at')
    print('qbits=32).  basic overhead grows as N^3 qbits^2 / 1 -- no k-amortization here.')

if __name__ == '__main__':
    test()
