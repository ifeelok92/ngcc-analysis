#!/usr/bin/env python3
"""
Loong256 end-to-end decapsulation (structural r2 leak, same method as Loong128).

Differences from the Loong128 driver:
  - BKZ 2.0 (BKZ2Param) with mpfr float type and configurable precision:
    the 337-dim embedding lattice (m=240 equations, n=96 unknowns) crashes
    fplll's default double-precision Babai ("infinite loop in babai").
  - progressive schedule 20 -> 30 -> 40 to speed up convergence.
  - unbuffered stdout so a background log shows live progress.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

from fpylll import BKZ, IntegerMatrix, LLL

from recover_msg_from_r2 import block_negacyclic_row, bits_to_bytes, mat_mul
from recover_r2_u2 import build_equations, load_dump

FLOAT_TYPE = "mpfr"  # passed to BKZ.reduction(), not BKZ.Param
PRECISION = 212      # bits; high enough to avoid 'infinite loop in babai'


def center(x, q):
    x %= q
    return x - q if x > q // 2 else x


def embedding_attack_bkz2(A, b, secret, q, eta, block_size, m_limit, progress=None):
    """Kannan embedding solved with BKZ 2.0 in mpfr precision."""
    if m_limit:
        A = A[:m_limit]
        b = b[:m_limit]
    m, n = len(A), len(A[0])
    dim = m + n + 1
    B = IntegerMatrix(dim, dim)
    for i in range(m):
        B[i, i] = q
    for j in range(n):
        row = m + j
        for i in range(m):
            B[row, i] = center(A[i][j], q)
        B[row, m + j] = 1
    last = m + n
    for i in range(m):
        B[last, i] = center(b[i], q)
    B[last, last] = 1

    LLL.reduction(B)
    if block_size:
        for bs in (progress or [block_size]):
            if bs >= block_size:
                bs = block_size
            t0 = time.time()
            par = BKZ.Param(block_size=bs, max_loops=8)
            # float_type/precision are kwargs of BKZ.reduction itself, NOT Param:
            # 337-dim lattice aborts fplll's double-precision Babai otherwise.
            BKZ.reduction(B, par, float_type=FLOAT_TYPE, precision=PRECISION)
            print(f"[bkz] block={bs} done in {time.time()-t0:.1f}s", flush=True)
            if bs == block_size:
                break

    target = tuple(secret)
    candidates = []
    for r in range(min(dim, 200)):
        v = [B[r, c] for c in range(dim)]
        if abs(v[-1]) != 1:
            continue
        s = tuple(-x for x in v[m:m + n]) if v[-1] == 1 else tuple(v[m:m + n])
        if all(-eta <= x <= eta for x in s):
            res = [center(sum(ai * si for ai, si in zip(row, s)) - bi, q)
                   for row, bi in zip(A, b)]
            mx = max(abs(x) for x in res)
            rms = (sum(x * x for x in res) / len(res)) ** 0.5
            candidates.append((mx, rms, s))
        if s == target:
            print("[info] true r2 seen in reduced basis", flush=True)
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates


def diagonal_options(data, r2):
    n, k2, q = data["N"], data["K2"], data["Q"]
    delta = (q + 1) // 2
    r2mat = block_negacyclic_row(r2, n, k2, q)
    rb = mat_mul(r2mat, data["B"], n, k2 * n, n, q)
    z = [center(v - t, q) for v, t in zip(data["V"], rb)]
    all_options = []
    for d in range(n):
        entries = []
        for i in range(n):
            j = (i + d) % n
            sign = 1 if i + d < n else -1
            entries.append((i, j, sign, z[i * n + j]))
        seen = {}
        for p in range(q):
            score = 0
            chosen = []
            for _i, _j, sign, val in entries:
                e0 = center(val - sign * p, q)
                e1 = center(val - sign * p - delta, q)
                if abs(e0) <= abs(e1):
                    score += e0 * e0
                    chosen.append(0)
                else:
                    score += e1 * e1
                    chosen.append(1)
            tup = tuple(chosen)
            if tup not in seen or score < seen[tup]:
                seen[tup] = score
        opts = sorted(seen.items(), key=lambda kv: kv[1])[:2]
        all_options.append([bits for bits, _score in opts])
        print(f"[diag] diagonal {d + 1}/{n} done", flush=True)
    return all_options


def write_candidates(path, options, n):
    count = 0
    with open(path, "w", encoding="ascii") as f:
        for choice in itertools.product((0, 1), repeat=n):
            bits = [0] * (n * n)
            for d, pick in enumerate(choice):
                diag_bits = options[d][pick]
                for idx, i in enumerate(range(n)):
                    j = (i + d) % n
                    bits[i * n + j] = diag_bits[idx]
            msg = bits_to_bytes(bits)
            f.write("".join(f"{x:02x}" for x in msg) + "\n")
            count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("--verifier", required=True)
    p.add_argument("--m", type=int, default=240)
    p.add_argument("--block", type=int, default=40)
    p.add_argument("--precision", type=int, default=212)
    args = p.parse_args()

    data = load_dump(args.dump)
    print(f"params N={data['N']} K2={data['K2']} Q={data['Q']} ETA={data['ETA']}", flush=True)
    rows, rhs = build_equations(data)
    nvars = data["K2"] * data["N"]
    print(f"equations={len(rows)} unknowns={nvars} using={args.m} block={args.block}", flush=True)

    t0 = time.time()
    cands = embedding_attack_bkz2(rows, rhs, [99] * nvars, data["Q"], data["ETA"],
                                  args.block, args.m, progress=[20, 30, args.block])
    print(f"lattice phase: {time.time() - t0:.1f}s", flush=True)
    if not cands:
        raise SystemExit("r2 recovery failed")
    print(f"r2_candidates={len(cands)} best maxres={cands[0][0]} rms={cands[0][1]:.3f}", flush=True)
    r2 = list(cands[0][2])

    options = diagonal_options(data, r2)
    cand_path = Path(args.dump).with_suffix(".candidates")
    count = write_candidates(cand_path, options, data["N"])
    print(f"message_candidates={count}", flush=True)

    result = subprocess.run([args.verifier, args.dump, str(cand_path)],
                            check=False, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
