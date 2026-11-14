#!/usr/bin/env python3
"""Recompute all modular residuals for a candidate vector in JSON format."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('vector', help='a JSON list of 160 integers, or a result JSON containing candidates')
    p.add_argument('--system', default=str(Path(__file__).resolve().parent/'system_A.npz'))
    args = p.parse_args()
    data = np.load(args.system)
    f, y = data['F'], data['y']
    q, eta, n_poly = int(data['q']), int(data['eta']), int(data['N'])
    bound = 2*(eta + (8 if n_poly <= 16 else 4))
    raw = json.loads(Path(args.vector).read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        entries = raw.get('candidates', [])
        candidates = [c['x'] if isinstance(c, dict) else c for c in entries]
    elif raw and isinstance(raw[0], (int, float)):
        candidates = [raw]          # a bare integer list is one candidate vector
    else:
        candidates = raw            # a list of vectors
    results = []
    for index, x in enumerate(candidates):
        x = np.asarray(x, dtype=np.int64)
        if x.shape != (f.shape[1],):
            results.append({'index': index, 'length': int(x.size),
                            'expected_length': int(f.shape[1]),
                            'all_rows_pass': False})
            continue
        residual = (y-f@x+q//2) % q-q//2
        results.append({'index': index, 'max_abs_x': int(np.max(np.abs(x))),
                        'max_residual': int(np.max(np.abs(residual))),
                        'rms_residual': float(np.sqrt(np.mean(residual.astype(float)**2))),
                        'all_rows_pass': bool(np.max(np.abs(x)) <= eta and
                                              np.max(np.abs(residual)) <= bound)})
    print(json.dumps({'system': args.system, 'equations_checked': int(f.shape[0]),
                      'error_bound': bound, 'candidates': results}, indent=2))


if __name__ == '__main__':
    main()
