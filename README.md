# Crazy GSW: from eigenvectors to invariant subspaces

*A walk through GSW homomorphic encryption that starts from the thing GSW is
actually bad at -- efficiency -- and follows a single thread of
generalization: hide one eigenvalue (basic GSW), hide a list of eigenvalues
(packed GSW), hide a whole invariant subspace (matrix GSW). The correctness is
almost free at every step; the security is not, and exactly where it stops
being free is the whole story. The payoff, measured: at the same ciphertext
cost, matrix GSW computes a full `k x k` matrix product at an overhead of
`8 * qbits^2` mod-muls -- **independent of k** -- while basic GSW pays
`N^3 * qbits^2` and packed GSW pays `8 * k^2 * qbits^2`.*

**NOTE: this is a novel construction, still awaiting and welcoming serious cryptanalysis.**

Three reference implementations accompany this document, structured so that
diffing them shows only the meaningful changes (all three now share an
identical helper layer and `Params` record, and use fast `uint64` numpy
arithmetic throughout):

- `basic_gsw.py`  -- one secret, one scalar message (clean LWE, `k = 1`)
- `packed_gsw.py` -- `k` secrets, a vector message (SIMD / slot-wise)
- `matrix_gsw.py` -- `k` secrets, a `k x k` matrix message (matmul composition)

Each runs a tiny **TOY** config (correctness, instant) and a **FULL-ON** config
(`N = 2048`, effective LWE dim `k = 1024`) that actually times ciphertext
add/mul against the corresponding plaintext add/mul.

---

## 1. The problem: GSW is homomorphic, and GSW is wasteful

Here is GSW in one sentence:

**A GSW ciphertext is a matrix that has your secret key as an approximate
eigenvector, and the eigenvalue is the message.**

Write the secret as a vector `s`. A ciphertext is a matrix `C` satisfying

```
s . C  ~  m . (s . G)
```

where `m` is the message and `G` is a fixed public "gadget" matrix. Squinting
past `G`, this says `s . C ~ m . s` -- `s` is a left-eigenvector of `C` with
eigenvalue `m`. The ciphertext is built as

```
C = Bbar . R + M . G
```

where `Bbar` is a public key with `s . Bbar ~ 0` (the LWE relation), `R` is
fresh low-norm randomness, and `M` is a message matrix with `s . M = m . s`
(in the base case `M = m . I`, which trivially satisfies this for any `s`).

Eigenvalues are the *right* place to put the message because **eigenvalues
multiply**: if `s` is an eigenvector of `C1` (eigenvalue `m1`) and of `C2`
(eigenvalue `m2`), then

```
s . C1 . G^{-1}(C2)  =  m1 . s . G . G^{-1}(C2)  =  m1 . s . C2  ~  (m1 * m2) . (s . G)
```

so multiplying ciphertexts multiplies messages, for free, straight out of the
linear algebra. (Addition is even easier: eigenvectors are shared, eigenvalues
add.) The gadget is the one extra trick: `G^{-1}(C2)` is the *bit-decomposition*
of `C2` (a `{0,1}`-matrix with `G . G^{-1}(C2) = C2`), so the noise only ever
gets multiplied by tiny bits instead of `q`-sized entries, and it grows
polynomially per multiplication rather than exponentially.

So GSW is beautiful. It is also **spectacularly inefficient**. To carry a
single small integer `m`, basic GSW allocates an entire `N x W` ciphertext
matrix (with `W = N * qbits`), pays `N * W^2` modular multiplications per
homomorphic product, and stores `N * W * qbits` bits -- all to hide one
`msgbits`-bit value. The eigenvector picture makes the waste obvious: the
ciphertext lives in a high-dimensional space for security reasons, and we use
exactly **one degree** of its eigenstructure. The rest is dead weight.

This document is about reclaiming that dead weight -- and the surprising thing
that happens when you reclaim *all* of it.

---

## 2. Level 1 -- packed GSW: pack a list of eigenvalues

The inefficiency suggests an immediate idea: instead of one eigenvector `s`
carrying one eigenvalue `m`, use `k` secret eigenvectors `s_0, ..., s_{k-1}`
and a message *matrix* `M` such that

```
s_i . M  =  m[i] . s_i      for each i
```

i.e. `M` is *diagonal* with respect to the `s_i`. Build a single public key
`Bbar` that *all* the `s_i` annihilate (`s_i . Bbar ~ 0` for every `i`), form
the ciphertext exactly as before,

```
C = Bbar . R + M . G
```

and decrypt each slot with its own secret:

```
s_i . C  ~  m[i] . (s_i . G)
```

One ciphertext, `k` messages. The eigenvalue-passthrough still works *per
slot*: each `s_i` is simultaneously an eigenvector of `C1` and `C2`, so

```
s_i . C1 . G^{-1}(C2)  ~  (m1[i] * m2[i]) . (s_i . G)
```

-- slot-wise (Hadamard) multiplication, SIMD-style, exactly like BFV/CKKS but
in GSW's matrix language. Single GSW is the `k = 1` case. This is
`packed_gsw.py`.

### Two correctness gotchas (both instructive)

**Gotcha 1: you need a *known* reference to read the eigenvalue against.**
With the convention `s = (-r, 1)`, the bottom-row block of `s . G` is exactly
the gadget vector `(1, 2, 4, ...)` -- a *known constant, independent of the
secret*. That known reference is what makes readout possible. But if each
`s_i` is fully random, then `s_i . G` is random everywhere, and
`m[i] . (random vector) mod q` is undecodable. The fix: **pin a known
coordinate into each secret.** Give `s_i` a fixed `1` in a distinct known slot
(and `0` in the others), so `s_i . G` at that slot is the known gadget. Each
secret gets its own private "readout window."

**Gotcha 2: read a low gadget coordinate, not the top one.** The top gadget
coordinate `m * 2^{qbits-1}` wraps for any `m >= 2`. Read a coordinate
`m * 2^j` with `j` low enough to leave headroom (we read index
`qbits - msgbits - 1`), then reduce mod `2^msgbits`.

### The design philosophy of packed GSW, in one line

> The ciphertext matrix has lots of eigenstructure; basic GSW uses one
> eigenvector, so let's plant `k` of them and read `k` eigenvalues off the
> same matrix.

### The assumption, honestly: it IS LWE -- `k` instances of it

Single GSW's security is **clean LWE**. The public key
`Bbar = [B ; r.B + e]` is, by the LWE assumption, indistinguishable from
uniform; `B` is random, the bottom row is a noisy linear combination that LWE
says looks random, and the whole thing hides `r`. One of the most well-studied
assumptions in cryptography.

The packed scheme plants `k` secret rows `s_0 .. s_{k-1}` and needs
`s_i . Bbar ~ 0` for all `i`. With the brave layout `dim = 0`, the secret is
`S = [solve | I_k]` and the public key splits as `Bbar = [B_solve ; B_pin]`
with the annihilation relation reading

```
    solve . B_solve + B_pin = E        i.e.  B_pin = -solve . B_solve + E
```

Strip the public `I_k` pin (it is just a bookkeeping identity, not secret) and
**look only at the sub-matrix `(B_solve, B_pin)`**: row `l` gives
`B_pin[l,:] = (-solve[l,:]) . B_solve + E[l,:]`, which is *exactly an LWE
instance* with secret `solve[l,:]` (a `k`-dimensional vector), samples
`B_solve` (a `k x W` random matrix), and noise `E[l,:]`. So the whole public
key is **`k` independent LWE instances** (one per row of `solve`), each with
secret dimension `k = N/2`, **sharing the same `W` samples `B_solve`**.

Two things to notice, both reassuring:

1. **The pins do not weaken security.** `I_k` is public, like the constant `1`
   in basic GSW's secret `s = (-s0, 1)` (that `1` is the affine term of plain
   LWE -- nobody counts it against you). The `k^2` pin entries are "handed to
   you for free" precisely because they are *not secret*; they do not reduce
   the unknowns the attacker must recover. The effective secret dimension is
   `k = N/2` (the `solve` block), **not `N`**.

2. **Shared samples do not weaken security.** The samples `B_solve` are public
   in plain LWE anyway -- the attacker already sees them. Giving the same
   `B_solve` to all `k` instances reveals nothing extra, and the other rows
   `B_pin[l' != l]` involve *different* unknowns `solve[l':]`, so they give no
   information about `solve[l,:]`. Formally: matrix-LWE `(A, S.A + E)` with
   `S` a `k x k` secret is equivalent to vector-LWE with a `k`-dimensional
   secret and `W` samples (embed a vector challenge as one row; conversely each
   row is an independent vector-LWE instance).

So the **only** difference between basic GSW and packed/matrix GSW is:

> basic GSW: 1 noisy null vector of `Bbar`, secret dim `dim = N-1`.
> packed/matrix GSW: `k` noisy null vectors, each secret dim `k = N/2`.

It is "distinguish given 1 noisy null vector" vs "distinguish given `k` noisy
null vectors" -- and the `k` vectors are *independent* LWE instances. There is
no bespoke assumption: **packing costs you `N/2` of dimension per instance but
buys you `k` message slots; it stays inside LWE.** (We are glossing over the
fact that `dim=0` makes the per-instance secret `k = N/2`, half of basic's
`N-1` -- that is the real security tax of brave packing, and it is why `N` must
be sized off `k = N/2`, not `N`; see the parameter discussion below.)

### Keygen has no matrix inverse (null-space coset sampling)

The annihilation relation `S . Bbar = E` says `Bbar` lives in the affine coset
`{B : S.B = E}` of the right null space of `S`. We sample `Bbar` *uniformly*
from that coset, **without ever inverting `solve`**. A basis `N` for the null
space of `S = [free | solve | I_k]` is built by construction from `S`'s entries:

```
   N = [  I_dim    0    ]   free rows       S . N = 0,  since
       [  0       I_k   ]   solve rows        free.I + solve.0 + pin.(-free) = 0
       [ -free   -solve ]   pin rows          0 + solve.I + pin.(-solve) = 0
```

Drawing `X = [X_free ; X_solve]` uniform and setting `Bbar = N . X + [0;0;E]`
gives `B_free = X_free`, `B_solve = X_solve`, `B_pin = E - free.X_free -
solve.X_solve`, and `S . Bbar = E`. For fixed `E` this is uniform over the
coset (parameterized by the free uniform `B_solve`) -- the **same**
distribution the old `B_solve = solve^{-1}(E - ...)` scheme produced, but that
old scheme needed `solve` *invertible* (odd determinant mod `2^qbits`) to
express `B_solve` through the inverse, and re-rolled until it found one. The
null-space version works for **any** `solve` (it makes `B_solve` the free
variable instead), so there is **no `matinv_mod`, no retry loop, no odd-det
check**. At `k = 1024` this turns keygen from a >20-minute Python-loop matrix
inversion into a 6-second pair of BLAS matmuls. (The `matinv_mod` is gone from
all three files; basic GSW never had one.)

### Sizing `N`: the effective dimension is `k = N/2`

Because the pins are public, the attacker faces `k` independent LWE instances
of secret dimension `k = N/2` (not one `N`-dimensional instance). So `N` must
be sized off `k`, the way standard LWE is sized off `n`. The reference points
(standard, no ring):

| scheme        | `n`    | `qbits` | security |
|---------------|--------|---------|----------|
| Frodo-640     | 640    | 16      | ~128-bit |
| Frodo-976     | 976    | 16      | ~192-bit |

LWE hardness grows with `n` and shrinks with `log q`; roughly `n` must scale
with `qbits` for fixed security. Our `qbits = 32` is twice Frodo's `16`, so to
match Frodo-640's ~128-bit we want `k` in the low thousands. We run **`N = 2048`,
`k = 1024`**: effective dimension `1024` at `qbits = 32`, plausibly ~100-128-bit
*conjectured* (this is a toy codebase for pedagogy, not a vetted parameter set).
`N = 4096` (`k = 2048`) would be more conservative but is time-infeasible in
numpy (~1 hour per ciphertext mul: the GEMM is `8.8e12 * 8` flops). `N = 2048`
is the time-feasible conjectured-secure size, and because matrix GSW's op-count
overhead `8 qbits^2` is `N`-independent, the *ratio* we measure there carries
to any `N`; only the wall-clock scales as `N^3`.

---

## 3. Aha -- we can take it further

Packed GSW plants `k` eigenvectors and reads `k` eigenvalues. But look at the
algebra we actually run: decryption computes

```
S . C  ~  diag(m) . (S . G)
```

and the *only* reason `diag(m)` is diagonal is that we forced `M` to be
diagonal so each `s_i` would be an eigenvector. The pin-trick readout,
though, does not actually need `M` to be diagonal. It needs something much
weaker: that `(s_j . G)` at pin block `p_l` is `delta_{jl} . g`. That fact is
purely about the *secret layout* (`pin = I_k`), not about `M` at all.

So suppose `m` is a **full `k x k` matrix**, not a diagonal, and require merely

```
S . M  =  m . S
```

-- no diagonal restriction. The rows of `S` are no longer eigenvectors of `M`;
instead, the row space of `S` is an **`m`-invariant subspace**. We have gone
from hiding a list of eigenvectors to hiding a subspace. Decryption still
works, because

```
S . C  ~  m . (S . G)
```

and row `i` of `m . (S . G)`, at pin block `p_l`, is

```
sum_j  m[i,j] * (s_j . G)[p_l]  =  sum_j  m[i,j] * delta_{jl} . g  =  m[i,l] . g
```

-- so we read `k^2` values (one per `(row i, pin l)` pair) instead of `k`, and
recover the entire matrix `m`.

And multiplication composes by **ordinary matrix product**:

```
S . (C1 . G^{-1}(C2))  =  m1 . S . G . G^{-1}(C2)  =  m1 . S . C2  ~  (m1 @ m2) . S . G
```

so `mul_cts(ct1, ct2)` encodes `m1 @ m2` -- non-commutative, real matmul.
Diagonal `m` recovers packed GSW (slot-wise product); `k = 1` recovers basic
GSW. This is `matrix_gsw.py`.

### The design philosophy of matrix GSW, in one line

> Eigenvalues were the wrong abstraction -- they forced `M` to be diagonal.
> Invariant subspaces let `M` be arbitrary, so one ciphertext carries a whole
> matrix, and ciphertext multiplication becomes matrix multiplication.

### The security price: *nothing new* (matrix = packed, exactly)

Here is the crucial point. The keygen and public key in `matrix_gsw.py` are
**byte-for-byte identical** to `packed_gsw.py`. The secret layout
`[free | solve | pin]`, the null-space coset keygen, the `k`-independent-LWE
assumption -- all unchanged. The message lives in `M . G`, re-randomized by
`Bbar . R` with fresh low-norm `R`, exactly as in packed/standard GSW. Growing
the message space from `k` (a vector) to `k^2` (a matrix) introduces **no new
structural leakage**: the only ciphertext-level change is that `M` is now
built from `S . M = m . S` instead of `S . M = diag(m) . S`, which is a
property of the *message encoding*, not of the *public structure* we publish.
So matrix GSW sits on exactly the same `k`-instance LWE assumption as packed
GSW. It is a free lunch *given that you already accepted packed GSW's lunch*.

---

## 4. Properties: the overhead, derived and measured

The whole point was efficiency. So price it concretely.

### Being brave: `k = N/2`

The secret layout is `N = dim + 2k` (`[free | solve | pin]`, with `solve` and
`pin` each `k` wide). The bravest packing sets **`dim = 0`**, so `k = N/2` --
half the ciphertext width is message slots, and there is no "wasted" free
block at all. (Correctness is unaffected: with `dim = 0` the secret is just
`S = [solve | I_k]` and the keygen samples `Bbar` from the null-space coset
`{B : S.B = E}` as above.) Security-wise this is the most structured variant:
the effective LWE dimension drops to `k = N/2` (the `solve` block; the `I_k`
pins are public), so it gives up the `dim` columns of hidden secret entropy.
That is the real tax of brave packing -- not "leaving LWE" (it stays LWE, see
§2) but halving the per-instance dimension -- and it is why `N` must be sized
off `k = N/2`.

### The overhead formula

Homomorphic multiplication is `mul_cts(C1, C2) = C1 . G^{-1}(C2)`, with `C1`
of shape `N x W` and `G^{-1}(C2)` of shape `W x W` (`W = N * qbits`). It costs

```
cost(ciphertext mul)  =  N * W^2  =  N^3 * qbits^2      mod-muls, mod 2^qbits.
```

The plaintext operation it imitates costs

```
cost(plaintext op)  =  1        (basic: 1 scalar mul)
                    =  k        (packed: k slot-wise muls)
                    =  k^3      (matrix: k x k matmul)
```

Now plug in `k = N/2` (i.e. `N = 2k`):

```
                                    ciphertext mul     plaintext op    OVERHEAD
basic   (k = 1, so N = 2):          8 * qbits^2        1               8 * qbits^2
packed  (k = N/2):                  8 * k^3 * qbits^2  k               8 * k^2 * qbits^2
matrix  (k = N/2):                  8 * k^3 * qbits^2  k^3             8 * qbits^2
```

Two things jump out.  But first, the question everyone asks: **where does the
`qbits^2` come from, and shouldn't it be divided by the plaintext modulus?**

### Where `qbits^2` comes from

The ciphertext mul is `C1 . G^{-1}(C2)`, with `C1` of shape `N x W` and
`G^{-1}(C2)` of shape `W x W`, where `W = N * qbits`.  That `qbits` factor in
`W` is the **gadget expansion**: each of the `N` ciphertext rows is stretched
by a factor of `qbits` into the powers-of-2 gadget columns (the
bit-decomposition that lets GSW multiply by a full-size value while only ever
scaling the noise by tiny `{0,1}` bits).  A matmul `N x W` by `W x W` pays
`N * W^2 = N * (N*qbits)^2 = N^3 * qbits^2`, so the `qbits^2` is the gadget
expansion showing up in **both** the row and the column inner dimension of the
matmul.  With `k = N/2` that is `8 * k^3 * qbits^2`, and the `k^3` cancels a
`k x k` matmul's `k^3` to leave `8 * qbits^2`.  So `qbits^2` is the price of
GSW's noise-controlled multiplication -- it has nothing to do with the message.

### Does the plaintext modulus divide it?  (three honest notions of overhead)

No -- not for the *time* overhead, and this is the subtle point.  There are
three natural notions, and the plaintext modulus enters only one of them:

**1. Op-count overhead (the wall-clock-fair metric, and the one FHE papers
use).**  Count each modular multiplication as one unit, regardless of the
modulus it is done mod.  This is the fair CPU metric **when both operands fit
in a machine word**: a mod-mul mod `2^qbits` and a mod-mul mod `2^msgbits` are
*each one MUL instruction* on a 64-bit CPU as long as `qbits, msgbits <= 64`.
A smaller plaintext modulus does **not** make the plaintext op cheaper than one
instruction -- one instruction is the irreducible floor -- so the plaintext
modulus does NOT appear in this ratio:

```
overhead_op  =  8 * qbits^2            (matrix, k = N/2; NO msgbits)
```

This is `8,192x` at `qbits = 32`, and it is the number the scripts print as
`op-count`.  This is the right overhead to quote on a word machine.

**2. Bit-operation overhead (schoolbook; the information-theoretic asymptotic).**
If you refuse the one-instruction abstraction and count raw bit-ops
(schoolbook mul of two `b`-bit numbers is `O(b^2)`), then a ciphertext mod-mul
on `qbits`-bit values costs `qbits^2` bit-ops and a plaintext mod-mul on
`msgbits`-bit values costs `msgbits^2` bit-ops, so

```
overhead_bit  =  8 * qbits^2 * (qbits / msgbits)^2  =  8 * qbits^4 / msgbits^2
```

NOW the plaintext modulus is in the denominator -- but squared, and `qbits`
moves to the **fourth** power, so this is `8 * 32^4 / 8^2 = 131,072x` at
`qbits=32, msgbits=8` -- NOT `8 * qbits^2 / msgbits`.  (An earlier version of
this writeup wrote `8 * (qbits/msgbits)^2 = 128x` here; that was a plain bug --
it forgot the base `8 * qbits^2` factor and was even *smaller* than the op-count
8192x, which is absurd since weighting the ciphertext operands as bigger must
make the overhead bigger, not smaller.)  This metric only becomes the
wall-clock truth in the **multi-precision regime**, `qbits > word_size` (real
FHE uses `q ~ 2^128..2^256`), and even there the honest factor is
`(qbits / word_size)^2` applied to the *ciphertext* side, with the plaintext
stuck at one word -- so the plaintext modulus still does not enter; the machine
word size enters the numerator instead.

**3. Storage expansion (where the plaintext modulus genuinely divides).**  A
ciphertext is `N x W` entries of `qbits` bits = `N * W * qbits = N^2 * qbits^2`
bits; the message it carries is `k^2 * msgbits` bits (matrix).  The ratio of
ciphertext bits to useful message bits is

```
storage  =  N^2 * qbits^2 / (k^2 * msgbits)  =  4 * qbits^2 / msgbits      (matrix, k = N/2)
```

-- and HERE the plaintext modulus is in the denominator, linearly: `512x` at
`qbits=32, msgbits=8`.  This is a bandwidth/density notion, not a time notion:
how many ciphertext bits you ship per plaintext bit of useful work.

So the answer to "shouldn't it be divided by the plaintext modulus?" is:
**yes for storage, no for wall-clock time.**  The time overhead a word CPU
actually pays is the op-count `8 * qbits^2`, with no `msgbits` in it, because a
small-plaintext multiply is still one instruction.  The scripts print all three
so you can see them side by side.

### Back to the comparison

**(a) Matrix GSW's time overhead is `8 * qbits^2`, independent of `k` and `N`.**
The `k^3` in the ciphertext-mul cost exactly cancels the `k^3` in a `k x k`
matrix product.  Packing is **asymptotically free**: you carry a whole `k x k`
matrix at the same per-multiplication op-count overhead as a single scalar.

**(b) Packed GSW is `k^2` times worse than matrix GSW.** Packed and matrix
share the *identical* ciphertext (same `N`, `W`, `qbits`, same keygen, same
`C`, same `8 k^3 qbits^2` mul cost), but packed only does `k` slot-wise muls
out of the `k^3` the ciphertext mul could have done.  So packed's op-count
overhead `8 k^2 qbits^2` is exactly `k^2` bigger.  Packed GSW wastes a factor
of `k^2` relative to what the same ciphertext could carry.  This is the real
argument for matrix GSW: **the expensive part is the ciphertext multiplication,
and that cost depends only on the key dimension, not on how much message you
pack into it.**  Matrix GSW is the move that maximizes useful plaintext work
for a fixed ciphertext cost.

### Measured: a concrete runtime test

Each script runs a FULL-ON config at **`N = 2048`** (`qbits = 32`; basic uses
`dim = 2047, k = 1`, packed/matrix use `k = 1024, dim = 0`) and times ciphertext
add/mul against the corresponding plaintext add/mul. `N = 2048` gives effective
LWE dimension `k = 1024` at `qbits = 32` -- a real, conjectured-secure size
(see the sizing discussion above; comparable in spirit to Frodo-640's `n=640`
at `qbits=16`). Every matmul goes through BLAS (single float64 GEMM when the
product fits in 53 bits, else 16-bit limb-split into 4 BLAS GEMMs), and the two
`W x W`-sized operations (the encryption pad `R` and `G^{-1}(C2)`) are processed
in `N_BLOCKS = 16` column-chunks so peak memory stays ~9 GB instead of the
~38 GB a full `W x W` array would need. Both `basic` and `optimized` (blocked)
versions of `encrypt` / `mul_cts` are run at toy size (both must decrypt
correctly); only the optimized one runs at full width. No slow uint64 path runs
at full width.

Measured (same `N = 2048`, `qbits = 32` for all three; `msgbits = 8` for
full-on). Three theoretical overheads per the three notions above, plus the
raw wall-clock:

| scheme  | `k`   | plaintext op      | ct_mul   | pt_mul    | **op-count** | bit-op    | storage | wall-clock |
|---------|-------|-------------------|----------|-----------|--------------|-----------|---------|------------|
| basic   | 1     | 1 scalar mul      | 100.2 s  | 45 ns     | **8.8e12x**  | 1.4e14x   | 5.4e8x  | 2.2e9x     |
| packed  | 1024  | 1024 slot muls    | 99.4 s   | 2.93 us   | **8.6e9x**   | 1.4e11x   | 5.2e5x  | 3.2e7x     |
| matrix  | 1024  | 1024x1024 matmul  | 104.0 s  | 15.2 ms   | **8,192x**   | 131,072x  | 512x    | 6,722x     |

Read the table horizontally and the whole story is there:

- **All three pay the same ciphertext-mul cost** (~100 s) -- because they share
  `N = 2048`, `qbits = 32`, hence the same `N * W^2 = 8.8e12` mod-muls. The
  ciphertext does not care how much message you pack.
- **The useful plaintext work differs by `1 : 1024 : 1024^3 ~ 1e9`.** That is
  `1 : k : k^3`. So the op-count overhead falls from 8.8 trillion (basic) to
  8.6 billion (packed) to **8,192 (matrix)** -- a `k^2 = 1,048,576x` win of
  matrix over packed, and a `k^3 ~ 1e9x` win over basic, at identical ciphertext
  cost. Matrix GSW's `8,192 = 8 * 32^2` matches the formula exactly, and is the
  **same number at every `N`** -- the overhead is N-independent.
- **The plaintext modulus enters only the `bit-op` and `storage` columns**, not
  `op-count` -- exactly as the three-notions discussion predicted. `storage`
  (where `msgbits` divides linearly) drops from 5.4e8x (basic, 1 msg bit) to
  512x (matrix, `k^2 = 1,048,576` msg bits packed): the message-density win.
  `bit-op` (where `msgbits^2` divides but `qbits^4` dominates) is the schoolbook
  asymptotic and only becomes wall-clock reality in multi-precision.
- **Wall-clock now tracks op-count.** Both the ciphertext matmul *and* the
  plaintext matmul run on BLAS (the message entries are small, so the `k x k`
  plaintext product also fits in float64's 53-bit mantissa), so per-op
  throughput is comparable and the matrix wall-clock ratio **6,722x** sits just
  *below* the op-count ratio **8,192x** -- the gap is gone. (At `N = 512` the
  plaintext was still on slow uint64 and the ratio was 383x, an artifact; the
  BLAS-plaintext fix closed it.) The op-count `8 * qbits^2` remains the
  hardware-fair metric; wall-clock scales as `N^3` at larger `N` while op-count
  stays flat.

So the concrete payoff at a conjectured-secure LWE size: **at the cost of one
~100-second ciphertext multiplication, matrix GSW delivers a full
`1024 x 1024` matrix product; basic GSW spends the same ~100 seconds on a
single scalar multiplication, and packed GSW spends it on 1024 independent
scalar products.** The op-count overhead `8 * qbits^2 = 8,192x` is the same at
any `N`; only the ~100 s wall-clock scales as `N^3`.

### Caveats on the numbers

These are raw timings at `N = 2048, qbits = 32` (effective LWE dim `k = 1024`,
a conjectured-secure but not vetted size), with a deliberately tiny
`base_error = 1` and `msgbits = 8` chosen so the noise floor and message slots
are easy to read, not for conservative security. The `N^3 * qbits^2` ciphertext
cost is real; what scales it up in production is larger `qbits` (multi-precision,
where the limb-split/BLAS trick no longer applies and the schoolbook `bit-op`
column takes over) and deeper circuits (noise growth forcing larger `qbits`).
The `k`-amortization argument for matrix GSW gets *stronger* at larger `N`:
matrix overhead stays `8 qbits^2` while
packed's `8 k^2 qbits^2` and basic's `N^3 qbits^2` both grow. The point of the
table is the *structure* of the ratio, confirmed by measurement at full LWE
width: ciphertext-mul cost is set by the key, plaintext work by the message
shape, and matrix GSW maximizes the latter for a fixed former.

---

## 4b. Exploiting the `{0,1}` right factor: subset sums, not multiplications

There is one more speedup, and it comes straight from looking at what
`mul_cts` actually computes:

```
mul_cts(C1, C2)  =  C1 . G^{-1}(C2)
```

`G^{-1}(C2)` is the bit-decomposition, so **every entry on the right is `{0,1}`**.
Each output entry is therefore

```
out[i,j]  =  sum_l  C1[i,l] * B[l,j]      with B[l,j] in {0,1}
          =  sum of the rows of C1 selected by column j of B    -- a SUBSET SUM,
                                                                no multiplications.
```

A general matmul kernel does not know `B` is binary, so it does `N * W^2`
full multiplications, almost all of which are multiply-by-0-or-1 -- wasted
work. Two ways to exploit this, one asymptotic and one concrete.

### Asymptotic: method of four Russians  ->  `O(N W^2 / log W)`

Block the inner index `l` into chunks of `t` bits. For each chunk `c`, the
`t` bits of `B` pick one of `2^t` subsets of the corresponding `t` columns of
`C1`; precompute all `2^t` subset sums once per chunk (Gray-code DP, **only
additions, no multiplications**), then each output entry is a sum of `W/t`
*table lookups* instead of `W` multiply-adds:

```
precompute :  (W/t) * N * 2^t   additions      (one pass over the t bits per chunk)
gather     :  (W/t) * N * W     additions      (sum W/t table lookups per output)
total      ~  2 N W^2 / log W   (choosing t ~ log W)
```

That is the `log W` factor the question predicted -- a real asymptotic win,
exactly because the right factor is `{0,1}` so the work is subset-sum, not
multiply. `bench_mul.py` implements it (`four_russians`, Gray-code subset DP +
numpy fancy-index gather) and confirms correctness.

### Concrete: the `{0,1}` structure unlocks BLAS  ->  ~420x in numpy

The asymptotic win is real, but in *concrete* numpy it loses to a simpler
consequence of the same observation. Because `B` is `{0,1}`, each output entry
is a sum of at most `W` values each `< 2^qbits`, so the true integer result is
`< W * 2^qbits = N * qbits * 2^qbits`. For our params that is
`4096 * 2^32 = 2^44 < 2^53` -- the subset sums fit **exactly in `float64`'s
53-bit mantissa**. So we can compute the whole matmul in `float64` and let
BLAS (which numpy has for float but *not* for `uint64`) handle it:

```
fast_mul(C1, B)  =  uint64( float64(C1_centered) @ float64(B_centered) ) % q
```

(The `_to_signed` centering keeps the magnitude at `q/2` so the 53-bit bound is
met with room to spare; results are bit-identical to the `uint64` path because
the float sum is an exact integer.) This is not a new algorithm -- it is the
*same* `N * W^2` matmul -- but it runs on a hand-tuned BLAS GEMM kernel at
~34 GFLOPS instead of numpy's software `uint64` path at ~8e7 ops/s.

`bench_mul.py` measures all three. To avoid running any slow path at full width,
it uses three regimes: SMALL (`N = 32`) for correctness of every method vs an
exact object-dtype reference (and the `uint64` baseline, instant there); MEDIUM
(`N = 128, W = 4096`) for the timed comparison including four Russians; and
FULL (`N = 512, W = 16384`) for **only** the BLAS-fast methods (no `uint64`, no
object reference, no four Russians -- those are minutes-to-days at this size).
The MEDIUM timings:

| method                  | time     | ops/s      | correct |
|-------------------------|----------|------------|---------|
| (a) `uint64 @` (SMALL)  | 87 ms    | 8.1e7      | yes     |
| (b) `float64` BLAS      | 0.158 s  | 1.4e10     | yes     |
| (c) four Russians t=8   | 1.22 s   | 2.4e8      | yes     |

At FULL width (`N = 512`): single-float BLAS = **3.2 s**, limb-split 4xBLAS =
5.0 s (both ~3e10 ops/s). The pre-BLAS `uint64` baseline would be ~28 min and
is never run at this size.

So: **four Russians does 7.5x fewer operations than BLAS** (the `log W`
asymptotic, exactly as predicted) **but runs at ~60x lower per-op throughput**
(gather/scatter in Python/C-indexed numpy, not a BLAS GEMM), so it loses
concretely by ~8x. Four Russians would win concretely only with a hand-tuned
SIMD or GPU gather kernel that runs the lookups at near-GEMM throughput; in
plain numpy, the `{0,1}` structure's biggest practical gift is that the subset
sums fit in `float64`, which unlocks BLAS for a ~500x win over `uint64`.

### Integration

All three scheme scripts use the BLAS path automatically: their shared
`matmul` helper centers both operands to signed `[-q/2, q/2)`, checks
`W * max|A| * max|B| < 2^53`, and if so computes in a single `float64` BLAS
GEMM, otherwise **splits both operands into 16-bit limbs and does 4 BLAS GEMMs**
(`LL + 2^16 (LH+HL)` mod `2^qbits`, with `2^32 HH` vanishing). This covers
every matmul in the schemes with NO `uint64` matmul path at all:
`C1 . G^{-1}(C2)` (rhs `{0,1}` -> single GEMM), `Bbar . R` (rhs `{0,+-1}` ->
single GEMM), and the full-range `M . G`, `S . ct` (limb-split).

**Memory: blocking the `W x W` arrays.** The two `W x W`-sized objects -- the
encryption pad `R` and the bit-decomposition `G^{-1}(C2)` -- would be 34 GB
apiece at `N = 2048`. Each scheme therefore provides a `basic` (clear, full
`W x W`) and an `optimized` (blocked) version of `encrypt` and `mul_cts`:
`encrypt_opt` / `mul_cts_opt` process the right factor in `N_BLOCKS = 16`
column-chunks, generating and consuming one `W x (W/16)` slice at a time
(`_gadget_inv_cols` bit-decomposes a column-slice; `_random_low_norm_float`
generates `R` in slices). Column-blocking is exact -- `G^{-1}(C2)[:, cols] ==
_gadget_inv_cols(C2, cols)` -- so the optimized result is bit-identical to the
basic one. Peak memory drops from ~38 GB to ~9 GB. Both versions are run at toy
size (both must decrypt correctly); only the optimized one runs at full width.

**Plaintext on BLAS too.** The message entries are small (`< 2^msgbits`), so a
`k x k` plaintext matmul also satisfies `k * 2^(2 msgbits) < 2^53` and runs on
BLAS. This is a fairness fix: with the ciphertext on BLAS and the plaintext on
slow `uint64`, the wall-clock ratio was artificially low (383x at `N = 512`).
With both on BLAS, the matrix wall-clock ratio (6,722x at `N = 2048`) sits just
below the op-count ratio (8,192x) -- the two metrics now agree, as they should.

At the FULL-ON `N = 2048` runs above, one ciphertext mul takes ~100 s and one
encrypt ~110 s, all on BLAS, peak ~9 GB, with no change to any decrypted result.

---

## 5. The moral, restated

- **Basic GSW** hides one eigenvalue. It is clean LWE (one noisy null vector,
  secret dim `N-1`). It is also wildly wasteful: a giant matrix to carry one
  integer, at `N^3 qbits^2` overhead.
- **Packed GSW** hides a list of eigenvalues by planting `k` eigenvectors. The
  correctness is almost forced, and the public key is `k` independent LWE
  instances (one per secret row) of dim `k = N/2` -- still LWE, just `k` of them
  sharing the samples. You recover only `k` useful muls from a `k^3`-capacity
  ciphertext, so overhead `8 k^2 qbits^2`, still growing with `k`.
- **Matrix GSW** drops the diagonal restriction and hides an invariant
  subspace: `S . M = m . S` for a full matrix `m`. Ciphertext multiplication
  becomes matrix multiplication, non-commutative. The keygen is identical to
  packed GSW (same `k`-instance LWE assumption, no matrix inverse -- the null-
  space coset construction), so it adds *no new* security cost -- and because
  the brave `k = N/2` packing makes the `k^3` in the ciphertext-mul cost cancel
  the `k^3` in a matmul, the overhead collapses to a **`k`-independent
  `8 qbits^2`**, exactly matching basic GSW's per-mul cost while doing `k^3`
  times more useful work. Measured: `8,192x` op-count at `qbits = 32`, flat in
  `k`; wall-clock `6,722x` at `N = 2048`, tracking the op-count now that both
  sides run on BLAS.

The pattern is the whole story of building things from lattices: the
homomorphic algebra is the easy 20% and almost generalizes itself; the hard
80% is understanding exactly what assumption each extra secret you plant into
the public structure lands you on. Here the answer is cleaner than it first
looks: the `k` pins are public constants (like basic GSW's `1`), the public
key is `k` independent LWE instances of dim `k = N/2`, and matrix GSW is a
pure efficiency win *inside* that assumption -- it turns packed's
`k^2`-growing overhead into a constant without touching the keygen or the
public structure. The tax of brave packing is that the effective dimension is
`k = N/2`, not `N` -- so `N` must be sized off `k`, the way standard LWE is
sized off `n`.
