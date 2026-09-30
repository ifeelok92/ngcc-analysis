# ngcc-analysis

Cryptanalysis of NGCC (Next-generation Commercial Cryptographic Algorithms)
post-quantum KEM submissions.

Currently: structural attacks on the LoongKEM (kem-18) family.

## Instances

| directory | parameter set | status |
|---|---|---|
| `instances/loong256/` | Loong256 | r2 recovery + message candidates; full decapsulation pipeline |

More parameter sets (e.g. Loong512) will be added under `instances/` as work
progresses.

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
