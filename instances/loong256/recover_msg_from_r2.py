#!/usr/bin/env python3

from __future__ import annotations

import argparse

from recover_r2_u2 import build_equations, embedding_attack, load_dump


def center(x, q):
    x %= q
    return x - q if x > q // 2 else x


def modq(x, q):
    return x % q


def negacyclic_matrix(poly, n, q):
    out = [0] * (n * n)
    for i in range(n):
        for j in range(n):
            if j >= i:
                out[i*n+j] = modq(poly[j-i], q)
            else:
                out[i*n+j] = modq(-poly[n+j-i], q)
    return out


def block_negacyclic_row(vec, n, k2, q):
    out = [0] * (n * k2 * n)
    for blk in range(k2):
        mat = negacyclic_matrix(vec[blk*n:(blk+1)*n], n, q)
        for i in range(n):
            for j in range(n):
                out[i * k2 * n + blk*n + j] = mat[i*n+j]
    return out


def mat_mul(a, b, rows, inner, cols, q):
    out = [0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            s = 0
            for k in range(inner):
                s += a[i*inner+k] * b[k*cols+j]
            out[i*cols+j] = s % q
    return out


def bytes_to_bits(bs):
    bits = []
    for b in bs:
        for i in range(8):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bits):
    out = []
    for i in range(0, len(bits), 8):
        v = 0
        for j, bit in enumerate(bits[i:i+8]):
            v |= bit << j
        out.append(v)
    return out


def recover_message(data, r2):
    n, k2, q = data["N"], data["K2"], data["Q"]
    delta = (q + 1) // 2
    r2mat = block_negacyclic_row(r2, n, k2, q)
    rb = mat_mul(r2mat, data["B"], n, k2*n, n, q)
    z = [center(v - t, q) for v, t in zip(data["V"], rb)]
    bits = [0] * (n * n)
    diag_scores = []
    for d in range(n):
        entries = []
        for i in range(n):
            j = (i + d) % n
            sign = 1 if i + d < n else -1
            entries.append((i, j, sign, z[i*n+j]))
        best = None
        for p in range(q):
            score = 0
            chosen = []
            for i, j, sign, val in entries:
                e0 = center(val - sign * p, q)
                e1 = center(val - sign * p - delta, q)
                if abs(e0) <= abs(e1):
                    score += e0 * e0
                    chosen.append(0)
                else:
                    score += e1 * e1
                    chosen.append(1)
            if best is None or score < best[0]:
                best = (score, p, chosen)
        diag_scores.append(best[0])
        for (i, j, _sign, _val), bit in zip(entries, best[2]):
            bits[i*n+j] = bit
    return bits_to_bytes(bits), diag_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("--m", type=int, default=90)
    p.add_argument("--block", type=int, default=40)
    args = p.parse_args()
    data = load_dump(args.dump)
    A, b = build_equations(data)
    cands, _ = embedding_attack(A, b, data["R2"], data["Q"], data["ETA"], args.block, args.m, 1)
    if not cands:
        raise SystemExit("r2 recovery failed")
    r2 = list(cands[0][2])
    msg, scores = recover_message(data, r2)
    truth = data["MSG"]
    print("r2_match", r2 == data["R2"])
    print("msg_match", msg == truth)
    print("msg", "".join(f"{x:02x}" for x in msg))
    print("truth", "".join(f"{x:02x}" for x in truth))
    print("max_diag_score", max(scores))


if __name__ == "__main__":
    main()
