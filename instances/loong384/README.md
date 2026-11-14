# LoongKEM-384 (kem-18) break

This directory contains a self-contained proof-of-concept for a full
decapsulation break of the LoongKEM Loong384 parameter set
($N=20$, $k_1=72$, $k_2=8$, $q=8191$, $\eta=3$, $d_u=11$, $d_v=4$).

Given only the public key and one ciphertext, the attack recovers the
ephemeral vector $r_2$ used during encryption, and then the shared secret —
verified byte-for-byte against the reference implementation's decapsulation
output (`SS_MATCH true`).

Observed on one machine (WSL2, x86-64, single core):

| stage | time |
|---|---|
| lattice reduction (progressive BKZ up to 55) | ~1 h |
| weighted BDD beam decoding (width 65536) | ~1 min |
| message-candidate enumeration ($2^{20}$) | ~35 min |
| FO re-encryption oracle search | ~2 h |

End-to-end from public data this is roughly **4 h** on one core. The
reproduction steps below do **not** re-run the 1 h BKZ: the reduced basis
was frozen to `source_checkpoint.json` at attack time, so Step 2 (decoding)
loads it and finishes in about a minute, and Step 3 (message + shared
secret) takes about 2.5 h.

## Background

The LoongKEM encryption outputs a ciphertext part

$$U = \mathrm{Block}(r_1 A_2) + R_2 A_4 + E_4 \pmod q$$

Notation: all computations are over $\mathbb{Z}_q$ (integers mod
$q = 8191$); $r_1$ is another short ephemeral vector of the encryption;
$A_2, A_4$ are public matrices ($A_4$ is a generic, unstructured
$160 \times 160$ matrix); $R_2$ is the block-anti-circulant expansion of
$r_2$ (each of the $k_2$ blocks of $r_2$ becomes a negacyclic matrix, as
defined by multiplication in $\mathbb{Z}_q[X]/(X^N+1)$); $E_4$ is the
encryption error ($\lVert E_4\rVert_\infty \le \eta$, plus compression
error); and $\mathrm{Block}(r_1 A_2)$ is the block-anti-circulant product
of the other ephemeral part $r_1$ with $A_2$. Rows inside each block are
shifts of one another, so differencing them cancels the structured term
exactly and leaves 3040 noisy linear equations in the 160 coefficients of
the short vector $r_2$ alone:

$$F x = y + e \pmod q, \qquad F \in \mathbb{Z}_q^{3040 \times 160}, \qquad \lVert e \rVert_\infty \le 14, \qquad x \in [-3,3]^{160}$$

Here $x$ is the coordinate representation of $r_2$, $F$ and $y$ are
derived from the public $A_4$ and $U$ by that differencing, and the
differencing doubles the error bound (two $E_4$ entries plus two
compression errors per equation).

## Solution outline

**Step 0 — problem shape.** The system above is the starting point. A
direct Kannan embedding (401+ dims) with BKZ-40/50 fails at this size: the
uSVP gap is beyond the reach of ordinary BKZ-40/50, and adding more rows
only makes the dimension worse.

**Step 1 — quotient projection: 160 → 32 unknowns.** Use the factorization

$$X^{20}+1 = (X^4+1)\,\Phi_{40}(X)$$

With the 4×20 fold matrix

$$P[e,j] = (-1)^{\lfloor j/4 \rfloor} \;\text{ if } j \bmod 4 = e, \;\text{ else } 0$$

which satisfies the identity

$$P\,C_{20}(v) = C_4(Pv)\,P$$

($C_n$ = n×n negacyclic matrix, i.e. multiplication by $v$ in
$\mathbb{Z}_q[X]/(X^n+1)$), the assembled one-sided identity is

$$P\,R_{20}(x)\,A_4 = R_4(P_B x)\,(P_B A_4), \qquad P_B = I_8 \otimes P$$

where $I_8$ is the 8×8 identity and $\otimes$ denotes the Kronecker
(product) matrix operation, so $P_B$ stacks one copy of $P$ per block
(shape 32×160). The 32-dim projection $a = P_B x$ can then be recovered
first (608-row system, 93-dim BKZ-20 embedding, minutes). Note $P_B A_4$
must be kept as a 32×160 linear form — folding $A_4$ on both sides to
32×32 is invalid for a generic $A_4$.

**Step 2 — 128-dim lift.** The projection fixes only signed sums of 5
coordinates. Fixing $a$, write

$$x = x_0 + K t$$

where $x_0 \in \mathbb{Z}^{160}$ places the 4 projection coordinates at
the head of each block (rest zeros), $K \in \mathbb{Z}^{160 \times 128}$
has columns that parameterize $\ker(P_B)$ (16 free variables per block,
128 total), and $t \in [-3,3]^{128}$ is the remaining unknown.
Substituting into the original 3040 equations gives a 128-dim system;
after solving for $t$, the head coordinates of $x_0$ must also be
re-checked against $[-3,3]$.

**Step 3 — projection-residual noise feedback.** Here $D$ denotes the
same row-difference operator as in the Background (subtract, per block,
the sign-corrected row 0 from row $i$ — the operation that cancels the
structured term); $R_{20}(x_0)$ is the block-anti-circulant expansion of
the offset vector $x_0$, same convention as $R_2$ in the Background.
$\mathrm{center}(\cdot)$ is the centered representative map: it replaces
each entry $v$ by $\big((v + \lfloor q/2\rfloor) \bmod q\big) - \lfloor q/2\rfloor$,
i.e. the representative in $[-4095, 4096]$. Since
$P P^{\mathsf T} = 5I$ (each row of $P$ has three $\pm 1$ entries),
$H = P^{\mathsf T}P/5$ is an orthogonal projection ($H^2 = H$,
$H^{\mathsf T} = H$). With the correct $a$,

$$g = \mathrm{center}\big(D\,(P^{\mathsf T}P\,(U - R_{20}(x_0) A_4))\big)$$

contains only noise (kernel directions satisfy $P R_{20}(Kt) = 0$, and the
structured term still cancels under $D$). Replacing the lift target by
$y_{\text{new}} = \mathrm{center}(y_l - \lfloor g/5 \rceil)$ — where $y_l$
is the lifted-system target from Step 2 — lowers the error variance to
$4/5$ (measured $\max|g| = 19$).

**Step 4 — weighted embedding + progressive BKZ.** Take 152 of the 3040
rows, weight $w = 2$, embedding constant $M = 8$, and shift the free
coordinates by their conditional means $\mu$ (per-coordinate rounding of
the projection entries, a standard conditional-mean estimate); this gives
a 281-dim embedding, reduced by progressive BKZ (20 → 25 → … → 55, one
tour each, MPFR-212) in about an hour.

**Step 5 — extract the weighted BDD lattice, wide-beam decoding.** From the
reduced basis, zero out the last coordinate using the row with
$\pm M$ in it and delete that row/column to obtain the 280-dim lattice

$$L_0 = \{(2t,\; F_s t + qz)\}$$

where $t$ ranges over the 128 free coordinates (scaled by the weight
$w=2$), $F_s$ is the 152-row selected subsystem, $z \in \mathbb{Z}^{152}$
absorbs the modulus, and $\mu$ is the per-coordinate conditional-mean
shift from Step 4; the target of the BDD instance is $(2\mu, y_s)$.
Decoding uses a wide beam search along the
Gram–Schmidt coefficients, window

$$s_i = \max\!\big(1, \min(16, \lceil 8/\sqrt{r_i} \rceil)\big)$$

where $r_i = \lVert b_i^* \rVert^2$ is the squared Gram–Schmidt norm of
level $i$, keeping the best 65536 states per level. On the
same lattice Babai's nearest plane fails (error norm 13027, $t$ out of
range) while the beam search hits (34.4, all rows pass) — the difficulty
is decoding, not just reduction. Decoding takes ~1 minute from the frozen
basis.

**Step 6 — message and shared secret.** With $r_2$ known,
$V - R_2 B$ decomposes into a negacyclic part plus the message with only a
per-diagonal sign ambiguity: keep the top-2 bit options per diagonal
($2^{20}$ combinations) and resolve them by FO re-encryption against the
ciphertext.

## Symbol index

| symbol | meaning |
|---|---|
| $q$ | modulus, 8191 |
| $N$, $k_1$, $k_2$ | ring degree and block counts (20, 72, 8 for Loong384) |
| $\eta$ | noise bound of encryption errors (3) |
| $r_1$, $r_2$ | the two short ephemeral vectors sampled during encryption |
| $A_2$, $A_4$, $B$ | public matrices ($A_4$ generic; $B$ = decompressed $B_2$) |
| $U$, $V$ | the two ciphertext matrices |
| $R_2$, $R_{20}(x)$, $R_4(\cdot)$ | block/negacyclic expansions of short vectors |
| $x$ | coordinate form of $r_2$ (160 dims, $[-3,3]$) |
| $F$, $y$, $e$ | the difference system, its target, and error ($\le 14$) |
| $P$ | 4×20 quotient-projection fold matrix |
| $P_B$ | per-block assembly of $P$ (32×160, Kronecker with $I_8$) |
| $a$ | 32-dim projection $P_B x$ |
| $x_0$, $K$, $t$ | lift offset (160), lift matrix (160×128), free unknowns (128) |
| $D$ | row-difference operator canceling the structured term |
| $H$ | orthogonal projection $P^{\mathsf T}P/5$ |
| $g$, $y_l$ | noise estimate and lifted-system target |
| $w$, $M$, $\mu$ | coordinate weight (2), embedding constant (8), mean shift |
| $F_s$, $y_s$, $z$ | selected 152 rows, their target, modulus-introducing vector |
| $L_0$ | 280-dim weighted BDD lattice |
| $r_i$, $s_i$ | squared Gram–Schmidt norm and beam window at level $i$ |

## Contents

All files, in alphabetical order.

```
denoised_lift.npz       the 128-dim lifted system produced by Steps 1–3 of
                        the Solution outline, with the noise-feedback
                        target already applied. Arrays inside:
                          F, y            3040x128 lifted system (F t = y + e)
                          K (160x128)     lift matrix, r2 = x0 + K t
                          x0 (160)        lift offset vector
                          original_F/y    the original 3040x160 system
                          q, eta, error_bound   parameters (8191, 3, 14)

instance.txt            test instance: N/PK/CT/SS/A4/B/U/V (fixed seed)
make_message_candidates.py   from r2: diagonal top-2 options -> 2^20 messages
progressive.py          helper: beam-search schedule used by the decoder
qary_embedding.py       helper: centering and q-ary embedding utilities
r2_solution.json        recovered r2 (160 integers in [-3,3])
recover_msg_from_r2.py  helper: negacyclic expansion / matrix ops
recover_r2_u2.py        helper: dump parser and equation builder
reference_result.json   decoding output from the recorded successful run
run_bounded.py          subprocess wrapper with a time limit
source_checkpoint.json  frozen BKZ-reduced 281-dim embedding basis
                        ("frozen" = the ~1 h BKZ-55 reduction was run once
                        at attack time and its output basis was saved; all
                        reproduction steps load this file instead of
                        re-running the reduction)
system_A.npz            the 3040x160 difference system F, y (from U)
verify_candidate.py     recompute all 3040 residuals for a candidate r2
verify_loong384         FO re-encryption oracle (prebuilt x86-64 ELF)
verify_loong384_candidates.c  oracle source; see below to rebuild
weighted_decode.py      extract the weighted BDD lattice and beam-decode it
kem_src/                Loong384 reference implementation (full)
dump_loong384_kem.c     instance generator (writes files like instance.txt)
```

## Requirements

Python 3 with `numpy` and `fpylll` (Linux, WSL or macOS):

```bash
pip install numpy fpylll
```

A C compiler (gcc/clang) only if you want to rebuild the oracle from
source (it is included prebuilt for x86-64 Linux; on Apple silicon,
rebuild it from the included source).

## Reproducing

### Step 1 — verify the recovered r2 (seconds)

```bash
python3 verify_candidate.py r2_solution.json
```

Expected: `max_residual 9 <= 14`, all 3040 rows pass. The system matrix has
full rank 160 over $\mathbb{F}_{8191}$, so a vector in $[-3,3]^{160}$
satisfying every row is unique — this is the encryption's ephemeral $r_2$,
not merely a candidate.

### Step 2 — re-derive the lattice solution (about 1 minute)

```bash
export OPENBLAS_NUM_THREADS=1
python3 run_bounded.py --seconds 600 --record decode_status.json \
  python3 weighted_decode.py --checkpoint source_checkpoint.json \
  --width 65536 --out decoded
python3 - <<'PY'
import json, pathlib
r = json.loads(pathlib.Path('decoded/result.json').read_text())
xs = {tuple(c['x']) for c in r['candidates']}
exp = tuple(json.loads(pathlib.Path('r2_solution.json').read_text()))
print('beam decode reproduces r2:', xs == {exp})
PY
```

Expected: `True` (all decoded candidates deduplicate to the same vector).
This re-runs the decoding from the frozen reduced basis; it does not
re-run the ~1 h BKZ. Notes: decoding needs the GSO in MPFR-212 precision
(set inside the script); the frozen basis and the lifted system are the
exact inputs of the recorded successful run.

### Step 3 — recover the message and shared secret (about 2.5 h total)

```bash
python3 make_message_candidates.py instance.txt r2_solution.json \
  --system system_A.npz --out msg_candidates.txt
./verify_loong384 instance.txt msg_candidates.txt
```

Expected oracle output:

```
FOUND checked=580038
MSG d6eb33f7b9aee4d521d2cb7ab1d6d78ae3abda57fe9a54beb73f05bd0cb517240378db39887b690c062bf673e676bda03a75
SS  5282d3da49e204dbf323ed950ad062b0c758b3662b625b22e4c5107121cd105117dac3bc7683cb8df0d4dcbc9b2fc070
SS_MATCH true
```

`SS_MATCH true` means the re-encryption of the recovered message equals the
challenge ciphertext byte-for-byte, and the key-derivation output equals the
shared secret recorded when the instance was created. The `SS` line in
`instance.txt` is only for this final comparison; no step before it reads it.

### Rebuilding the oracle from source (optional, ~10 s)

```bash
gcc -O2 -march=native -mpclmul -maes -mavx2 -mbmi2 -o verify_loong384 \
  verify_loong384_candidates.c kem_src/auxfunc.c kem_src/drng.c kem_src/poly.c \
  -I kem_src
```

Do not add `kem_src/KEM_Loong.c` to this build: the oracle source already
inlines the KEM interfaces it needs, and linking both definitions fails.
Adjust the SIMD flags to your CPU (drop `-mavx2 -mbmi2` on older machines).

### Regenerating the instance (optional)

```bash
gcc -O2 -march=native -mpclmul -maes -mavx2 -mbmi2 -o dump_loong384 \
  dump_loong384_kem.c kem_src/KEM_Loong.c kem_src/auxfunc.c kem_src/drng.c \
  kem_src/poly.c -I kem_src
./dump_loong384 0x42 > instance.txt
```
