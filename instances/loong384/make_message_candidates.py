#!/usr/bin/env python3
"""End-to-end verification: recover the message and shared secret from r2.

Given the 160-dim ephemeral vector r2 (as JSON), this script
  1. checks r2 against the U-relation difference system (5520/3040 rows),
  2. computes V - R2*B, keeps the top-2 bit options per diagonal,
  3. enumerates all 2^N combinations and writes them as hex message
     candidates for the FO re-encryption oracle (verify_loong384).

Usage:
    python3 make_message_candidates.py <instance.txt> <r2.json> \
        --system system_A.npz [--out msg_candidates.txt]
Then:
    ./verify_loong384 <instance.txt> msg_candidates.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def center(x, q):
    x %= q
    return x - q if x > q // 2 else x


def load_dump(path):
    data = {}
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == "N":
                data["N"] = int(parts[1])
                data["K1"] = int(parts[3])
                data["K2"] = int(parts[5])
                data["Q"] = int(parts[7])
                data["ETA"] = int(parts[9])
                data["DU"] = int(parts[11])
                data["DV"] = int(parts[13])
            elif key in {"PK", "CT", "SS"}:
                data[key] = parts[1]
            else:
                data[key] = [int(v) for v in parts[1:]]
    return data


def negacyclic_matrix(poly, n, q):
    out = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            if j >= i:
                out[i * n + j] = poly[j - i] % q
            else:
                out[i * n + j] = (-poly[n + j - i]) % q
    return out


def block_negacyclic_row(vec, n, k2, q):
    out = [0] * (n * k2 * n)
    for blk in range(k2):
        mat = negacyclic_matrix(vec[blk * n:(blk + 1) * n], n, q)
        for i in range(n):
            for j in range(n):
                out[i * k2 * n + blk * n + j] = mat[i * n + j]
    return out


def mat_mul(a, b, rows, inner, cols, q):
    out = [0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            s = 0
            for k in range(inner):
                s += a[i * inner + k] * b[k * cols + j]
            out[i * cols + j] = s % q
    return out


def bits_to_bytes(bits):
    out = []
    for i in range(0, len(bits), 8):
        v = 0
        for j, bit in enumerate(bits[i:i + 8]):
            v |= bit << j
        out.append(v)
    return out


def diagonal_options(data, r2):
    """For each diagonal of V - R2*B, keep the two best bit assignments."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump", help="instance file with N/PK/CT/SS/A4/B/U/V")
    p.add_argument("solution", help="JSON file with the recovered r2 vector")
    p.add_argument("--system", default="system_A.npz",
                   help="NPZ with the U-relation system (F, y)")
    p.add_argument("--out", default="msg_candidates.txt",
                   help="output file for hex message candidates")
    args = p.parse_args()

    data = load_dump(args.dump)
    x = json.loads(Path(args.solution).read_text(encoding="utf-8"))
    n_unknowns = data["K2"] * data["N"]
    assert len(x) == n_unknowns, f"expected {n_unknowns} coordinates"

    d = np.load(args.system)
    q = int(d["q"])
    res = (d["y"] - d["F"] @ np.asarray(x, dtype=np.int64) + q // 2) % q - q // 2
    mx = int(np.max(np.abs(res)))
    rms = float(np.sqrt(np.mean(res.astype(np.float64) ** 2)))
    print(f"[step1] residual over {d['F'].shape[0]} rows: max={mx} rms={rms:.4f}")
    assert mx <= 14, "r2 fails the U-system residual check"

    options = diagonal_options(data, x)
    count = write_candidates(args.out, options, data["N"])
    print(f"[step2] wrote {count} message candidates to {args.out}")
    print(f"[next ] ./verify_loong384 {args.dump} {args.out}")


if __name__ == "__main__":
    main()
