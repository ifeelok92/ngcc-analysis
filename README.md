# ngcc-analysis

Cryptanalysis of NGCC (Next-generation Commercial Cryptographic Algorithms)
post-quantum KEM submissions.

Currently: structural attacks on the LoongKEM (kem-18) family.

## Instances

| directory | parameter set | status |
|---|---|---|
| `instances/loong256/` | Loong256 | r2 recovery + message candidates; full decapsulation pipeline |
| `instances/loong384/` | Loong384 | **full decapsulation break**: r2 recovered via splitting-field projection + lift + noise feedback + beam decoding; FO oracle confirms SS_MATCH true |

More parameter sets (e.g. Loong512) will be added under `instances/` as work
progresses.

## LoongKEM parameters and the lattice dimensions they induce

Official LoongKEM parameter sets (spec table 2, $q = 8191$ for all sets):

| set | $\lambda$ | $N$ | $(k_1, k_2)$ | $\eta$ | $(d_b, d_u, d_v)$ |
|---|---|---|---|---|---|
| Loong128 | 128 | 12 | (48, 4) | 5 | (10, 10, 4) |
| Loong256 | 256 | 16 | (64, 6) | 4 | (10, 10, 10) |
| Loong384 | 384 | 20 | (72, 8) | 3 | (11, 11, 6) |
| Loong512 | 512 | 24 | (80, 10) | 2 | (11, 11, 4) |

The attack reduces the ciphertext equation
$U_2 = \mathrm{Block}(r_1 A_2) + R_2 A_4 + E_4$ to noisy linear equations in
the $k_2 N$ coefficients of $r_2$. Taking $m$ of those equations and applying
a Kannan embedding gives a lattice of dimension $m + k_2 N + 1$:

| set | unknowns $k_2 N$ | difference equations | typical embedding dim | status |
|---|---|---|---|---|
| Loong128 | 48 | 528 | 139 ($m$=90, BKZ-40) | **broken**, full decapsulation |
| Loong256 | 96 | 1440 | 337 ($m$=240, BKZ-40) | **broken**, full decapsulation |
| Loong384 | 160 | 3040 | 401 ($m$=240) — direct BKZ-40/50 **fails** | **broken** via splitting-field projection (see below) |
| Loong512 | 240 | 5520 | — | not attempted |

## Splitting-field projection (Loong384)

For Loong384 the direct embedding is out of BKZ reach, so the successful
attack in `instances/loong384/` shrinks the lattice first, using the
factorization $X^{20}+1 = (X^4+1)\,\Phi_{40}(X)$:

```text
r2 (160 unknowns, ring R = Z_q[X]/(X^20+1))
  │  splitting-field projection via X^4+1, fold matrix P (4x20)
  ▼
a = PB*x : 32 unknowns   → 93-dim embedding, BKZ-20, minutes
  │  lift x = x0 + K t over ker(PB)
  ▼
t : 128 unknowns         → 281-dim weighted embedding (= 128 + 152 + 1),
                            progressive BKZ-55, ~1 h; frozen basis
  │  extract 280-dim weighted BDD lattice, wide beam decoding
  ▼
r2 recovered (160)       → V - R2*B → 2^20 message candidates
  │  FO re-encryption oracle
  ▼
message + shared secret (SS_MATCH true)
```

With the projection the effective lattice work drops from 401 dims to
281 dims (and 93 dims for the projection stage), which is what puts
Loong384 back inside BKZ reach on a single core.

## Background

The LoongKEM encryption equations mix block-negacyclic terms with plain
matrix products:

```text
U2 = Block(r1*A2) + R2*A4 + E4
V  = Block(r1*b1) + R2*B2 + E5 + Delta*M
```

Entries that must agree inside each negacyclic block cancel the hidden
`Block(r1*A2)` term, leaving noisy linear equations in the short ephemeral
`r2` alone. After `r2` is recovered, `V - R2*B2` decomposes into a
negacyclic part plus the message, leaving only `2^N` message candidates
that FO re-encryption resolves. See each instance directory for parameters
and reproduction steps.

## Requirements

- Linux or WSL
- Python 3 with `fpylll`
- gcc (to build the dump/verifier helpers against the reference
  implementation)

## Instance: Loong256

Run from `instances/loong256/`:

```bash
PYTHONUNBUFFERED=1 python3 attack_loong256_kem.py \
  loong256_kem_dump_1.txt \
  --verifier ./verify_loong256_candidates \
  --m 240 \
  --block 40
```

Expected result:

```text
r2_candidates=1 best maxres=10 rms=4.324
message_candidates=65536
FOUND checked=130
SS_MATCH true
```

Observed runtime on WSL:

```text
BKZ-20: 267.5s
BKZ-30: 428.2s
BKZ-40: 1121.4s
lattice phase: 1826.5s
```

The run writes `loong256_kem_dump_1.candidates`. The `SS` line in the dump
is only for final validation; it is not used by the lattice attack.

## Instance: Loong384

Run from `instances/loong384/` (see its README for the full three-step
walkthrough, timing table, and how the frozen reduced basis is used):

```bash
# Step 1: verify the recovered r2 against the difference system (seconds)
python3 verify_candidate.py r2_solution.json

# Step 2: re-run the beam decoding from the frozen reduced basis (~1 min)
export OPENBLAS_NUM_THREADS=1
python3 run_bounded.py --seconds 600 --record decode_status.json \
  python3 weighted_decode.py --checkpoint source_checkpoint.json \
  --width 65536 --out decoded

# Step 3: enumerate 2^20 message candidates and run the FO oracle (~2.5 h)
python3 make_message_candidates.py instance.txt r2_solution.json \
  --system system_A.npz --out msg_candidates.txt
./verify_loong384 instance.txt msg_candidates.txt
```

Expected result:

```text
FOUND checked=580038
SS_MATCH true
```

Observed runtime on WSL (single core): progressive BKZ-55 lattice
reduction ~1 h (already frozen into `source_checkpoint.json`), beam
decoding ~1 min, candidate enumeration ~35 min, FO oracle ~2 h.
