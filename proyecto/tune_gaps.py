import numpy as np

from proyecto import blosum, metrics, sampling, sw_affine

GAP_OPENS = [9.0, 10.0, 11.0, 12.0, 13.0]
GAP_EXTENDS = [1.0, 2.0]

HELDOUT_SEED = 99


def main():
    dataset = sampling.build_dataset(per_bin=15, seed=HELDOUT_SEED)
    pairs = [p for _, band in dataset for p in band]
    print(f'Tuning on {len(pairs)} held-out pairs '
          f'(seed={HELDOUT_SEED}, disjoint parameters from the benchmark)\n')

    substs = [blosum.score_matrix_for(p.seq_x, p.seq_y) for p in pairs]

    print(f'{"open":>5} {"ext":>4} {"mean F1":>9}')
    results = []
    for gap_open in GAP_OPENS:
        for gap_extend in GAP_EXTENDS:
            scores = []
            for pair, subst in zip(pairs, substs):
                _, matches = sw_affine.smith_waterman_affine(
                    pair.seq_x, pair.seq_y, subst, gap_open, gap_extend)
                scores.append(metrics.prf1(set(matches), pair.ref)[2])
            mean_f1 = float(np.mean(scores))
            results.append((mean_f1, gap_open, gap_extend))
            print(f'{gap_open:5.0f} {gap_extend:4.0f} {mean_f1:9.4f}')

    best_f1, best_open, best_extend = max(results)
    print(f'\nBest: gap_open={best_open:.0f}, gap_extend={best_extend:.0f} '
          f'(mean F1 = {best_f1:.4f})')
    print(f'Re-run the benchmark with: '
          f'--gap-open {best_open:.0f} --gap-extend {best_extend:.0f}')


if __name__ == '__main__':
    main()
