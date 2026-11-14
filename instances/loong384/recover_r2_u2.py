#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math

from fpylll import BKZ, IntegerMatrix, LLL


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
            elif key in {"PK", "CT", "SS"}:
                data[key] = parts[1]
            else:
                data[key] = [int(x) for x in parts[1:]]
    return data


def r2_coeff_vector_for_entry(A4, N, K2, row, col):
    coeff = [0] * (K2 * N)
    width = K2 * N
    for blk in range(K2):
        for t in range(N):
            a = A4[(blk * N + t) * width + col]
            if t >= row:
                idx = blk * N + (t - row)
                sign = 1
            else:
                idx = blk * N + (N + t - row)
                sign = -1
            coeff[idx] += sign * a
    return coeff


def build_equations(data):
    N, K2, Q = data["N"], data["K2"], data["Q"]
    A4, U = data["A4"], data["U"]
    rows = []
    rhs = []
    for blk in range(K2):
        base_col = blk * N
        for i in range(N):
            if i == 0:
                continue
            for j in range(N):
                if j >= i:
                    d, sign = j - i, 1
                else:
                    d, sign = N + j - i, -1
                c1 = r2_coeff_vector_for_entry(A4, N, K2, i, base_col + j)
                c0 = r2_coeff_vector_for_entry(A4, N, K2, 0, base_col + d)
                rows.append([center(sign * a - b, Q) for a, b in zip(c1, c0)])
                rhs.append(center(sign * U[i * K2 * N + base_col + j] - U[base_col + d], Q))
    return rows, rhs


def candidate_score(A, b, s, q):
    res = [center(sum(ai*si for ai, si in zip(row, s)) - bi, q)
           for row, bi in zip(A, b)]
    return max(abs(x) for x in res), (sum(x*x for x in res)/len(res))**0.5


def embedding_attack(A, b, secret, q, eta, block_size, m_limit, M):
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
    B[last, last] = M
    LLL.reduction(B)
    if block_size:
        BKZ.reduction(B, BKZ.Param(block_size=block_size, max_loops=2))
    candidates = []
    target = tuple(secret)
    true_norm = None
    for r in range(min(dim, 120)):
        v = [B[r, c] for c in range(dim)]
        if abs(v[-1]) != M:
            continue
        s = tuple((-x for x in v[m:m+n])) if v[-1] == M else tuple(v[m:m+n])
        if all(-eta <= x <= eta for x in s):
            mx, rms = candidate_score(A, b, s, q)
            candidates.append((mx, rms, s))
        if s == target:
            true_norm = math.sqrt(sum(x*x for x in v[:-1]))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates, true_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("--m", type=int, default=90)
    p.add_argument("--block", type=int, default=40)
    p.add_argument("--M", type=int, default=1)
    args = p.parse_args()
    data = load_dump(args.dump)
    A, b = build_equations(data)
    r2 = data["R2"]
    print(f"equations={len(A)} unknowns={len(r2)} using={args.m} block={args.block}")
    residuals = [center(sum(ai*si for ai, si in zip(row, r2)) - bi, data["Q"])
                 for row, bi in zip(A, b)]
    print(f"true residual max={max(abs(x) for x in residuals)} rms={(sum(x*x for x in residuals)/len(residuals))**0.5:.3f}")
    candidates, true_norm = embedding_attack(A, b, r2, data["Q"], data["ETA"], args.block, args.m, args.M)
    print(f"candidates={len(candidates)} true_norm={true_norm}")
    for i, (mx, rms, cand) in enumerate(candidates[:5]):
        print(f"cand {i}: maxres={mx} rms={rms:.3f} match={list(cand)==r2}")
    ok = bool(candidates) and list(candidates[0][2]) == r2
    print(f"RECOVERY {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
