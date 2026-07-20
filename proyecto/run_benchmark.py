# coding=utf-8
"""Benchmark classical Smith-Waterman against DEDAL on Pfam seed pairs.

Usage:  venv/bin/python -m proyecto.run_benchmark [--per-bin 50]
Writes: proyecto/results.csv
"""

import argparse
import csv
import os
import time

import tensorflow_hub as hub

from proyecto import blosum, metrics, sampling, sw_affine

MODEL_URL = 'https://tfhub.dev/google/dedal/3'
OUT_PATH = os.path.join(os.path.dirname(__file__), 'results.csv')

FIELDS = ['family', 'name_x', 'name_y', 'len_x', 'len_y', 'pid', 'bin',
          'n_ref', 'sw_prec', 'sw_rec', 'sw_f1', 'sw_ms',
          'dedal_prec', 'dedal_rec', 'dedal_f1', 'dedal_ms']


def run_sw(pair, gap_open, gap_extend):
    """Align one pair with the classical baseline. Returns (metrics, ms)."""
    subst = blosum.score_matrix_for(pair.seq_x, pair.seq_y)
    start = time.perf_counter()
    _, matches = sw_affine.smith_waterman_affine(
        pair.seq_x, pair.seq_y, subst, gap_open, gap_extend)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return metrics.prf1(set(matches), pair.ref), elapsed_ms


def run_dedal(model, pair):
    """Align one pair with DEDAL. Returns (metrics, ms)."""
    start = time.perf_counter()
    pred = metrics.dedal_matches(model, pair.seq_x, pair.seq_y)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return metrics.prf1(pred, pair.ref), elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-bin', type=int, default=50)
    parser.add_argument('--gap-open', type=float, default=11.0)
    parser.add_argument('--gap-extend', type=float, default=1.0)
    parser.add_argument('--out', default=OUT_PATH)
    args = parser.parse_args()

    print('Building dataset from Pfam seeds...')
    dataset = sampling.build_dataset(per_bin=args.per_bin)
    for index, pairs in dataset:
        low, high = sampling.PID_BINS[index]
        print(f'  PID [{low:.1f}, {high:.1f}): {len(pairs):3d} pairs')
    total = sum(len(p) for _, p in dataset)
    print(f'  total: {total} pairs')

    print(f'\nLoading DEDAL from {MODEL_URL} ...')
    model = hub.load(MODEL_URL)

    # The first call traces the TF graph, which costs seconds. Discard it so
    # it does not land on whichever pair happens to be first.
    warm = dataset[0][1][0]
    metrics.dedal_matches(model, warm.seq_x, warm.seq_y)
    print('Warm-up done.\n')

    rows = []
    done = 0
    for index, pairs in dataset:
        for pair in pairs:
            (sw_p, sw_r, sw_f1), sw_ms = run_sw(
                pair, args.gap_open, args.gap_extend)
            (dd_p, dd_r, dd_f1), dd_ms = run_dedal(model, pair)
            rows.append({
                'family': pair.family, 'name_x': pair.name_x,
                'name_y': pair.name_y, 'len_x': len(pair.seq_x),
                'len_y': len(pair.seq_y), 'pid': round(pair.pid, 4),
                'bin': index, 'n_ref': len(pair.ref),
                'sw_prec': round(sw_p, 4), 'sw_rec': round(sw_r, 4),
                'sw_f1': round(sw_f1, 4), 'sw_ms': round(sw_ms, 2),
                'dedal_prec': round(dd_p, 4), 'dedal_rec': round(dd_r, 4),
                'dedal_f1': round(dd_f1, 4), 'dedal_ms': round(dd_ms, 2),
            })
            done += 1
            if done % 20 == 0:
                print(f'  {done}/{total} pairs')

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {len(rows)} rows to {args.out}')

    print(f'\n{"PID band":>12} {"n":>4} {"SW F1":>7} {"DEDAL F1":>9} {"gap":>7}')
    for index, (low, high) in enumerate(sampling.PID_BINS):
        band = [r for r in rows if r['bin'] == index]
        if not band:
            continue
        sw = sum(r['sw_f1'] for r in band) / len(band)
        dd = sum(r['dedal_f1'] for r in band) / len(band)
        label = f'[{low:.1f}, {high:.1f})'
        print(f'{label:>12} {len(band):4d} {sw:7.3f} {dd:9.3f} {dd - sw:+7.3f}')


if __name__ == '__main__':
    main()
