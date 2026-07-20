# coding=utf-8
"""Figures and summary tables from the benchmark results.

Usage: venv/bin/python -m proyecto.plots [--csv proyecto/results_tuned.csv]
"""

import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use('Agg')  # No display in this environment.
import matplotlib.pyplot as plt

from proyecto import sampling

HERE = os.path.dirname(__file__)
BAND_LABELS = ['<0.1', '0.1-0.2', '0.2-0.3', '>0.3']


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('pid', 'sw_f1', 'sw_prec', 'sw_rec', 'sw_ms',
                    'dedal_f1', 'dedal_prec', 'dedal_rec', 'dedal_ms'):
            row[key] = float(row[key])
        for key in ('bin', 'len_x', 'len_y', 'n_ref'):
            row[key] = int(row[key])
    return rows


def mean_ci(values):
    """Mean and half-width of a 95% CI (normal approximation)."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    stderr = statistics.stdev(values) / len(values) ** 0.5
    return mean, 1.96 * stderr


def plot_f1_vs_pid(rows, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for key, label, color, marker in (
            ('sw_f1', 'Smith-Waterman + BLOSUM62', '#c0392b', 'o'),
            ('dedal_f1', 'DEDAL', '#2471a3', 's')):
        means, errs = [], []
        for index in range(len(sampling.PID_BINS)):
            band = [r[key] for r in rows if r['bin'] == index]
            mean, err = mean_ci(band)
            means.append(mean)
            errs.append(err)
        ax.errorbar(range(len(BAND_LABELS)), means, yerr=errs, label=label,
                    color=color, marker=marker, capsize=4, linewidth=2)

    ax.set_xticks(range(len(BAND_LABELS)))
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel('Identidad de secuencia (PID) del alineamiento de referencia')
    ax.set_ylabel('F1 sobre pares de columnas alineadas')
    ax.set_title('Calidad de alineamiento vs. identidad de secuencia')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Wrote {out_path}')


def plot_time_vs_length(rows, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = [r['len_x'] * r['len_y'] for r in rows]
    ax.scatter(sizes, [r['sw_ms'] for r in rows], s=12, alpha=0.6,
               color='#c0392b', label='Smith-Waterman (Python puro)')
    ax.scatter(sizes, [r['dedal_ms'] for r in rows], s=12, alpha=0.6,
               color='#2471a3', label='DEDAL (transformer + SW)')
    ax.set_xlabel('Tamaño de la matriz DP  (len_x $\\times$ len_y)')
    ax.set_ylabel('Tiempo por par (ms)')
    ax.set_yscale('log')
    ax.set_title('Costo por par: DP dependiente de la longitud vs. costo fijo')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Wrote {out_path}')


def summary(rows):
    print(f'\n{"Banda PID":>10} {"n":>4} '
          f'{"SW F1":>14} {"DEDAL F1":>14} {"brecha":>8}')
    for index, label in enumerate(BAND_LABELS):
        band = [r for r in rows if r['bin'] == index]
        if not band:
            continue
        sw_m, sw_e = mean_ci([r['sw_f1'] for r in band])
        dd_m, dd_e = mean_ci([r['dedal_f1'] for r in band])
        print(f'{label:>10} {len(band):4d} '
              f'{sw_m:7.3f} ± {sw_e:.3f} {dd_m:7.3f} ± {dd_e:.3f} '
              f'{dd_m - sw_m:+8.3f}')

    print(f'\n{"Banda PID":>10} {"SW prec":>9} {"SW rec":>8} '
          f'{"DED prec":>9} {"DED rec":>8}')
    for index, label in enumerate(BAND_LABELS):
        band = [r for r in rows if r['bin'] == index]
        if not band:
            continue
        print(f'{label:>10} '
              f'{statistics.mean([r["sw_prec"] for r in band]):9.3f} '
              f'{statistics.mean([r["sw_rec"] for r in band]):8.3f} '
              f'{statistics.mean([r["dedal_prec"] for r in band]):9.3f} '
              f'{statistics.mean([r["dedal_rec"] for r in band]):8.3f}')

    sw_ms = [r['sw_ms'] for r in rows]
    dd_ms = [r['dedal_ms'] for r in rows]
    lens = [max(r['len_x'], r['len_y']) for r in rows]
    print(f'\nTiempos por par (n={len(rows)}, longitudes {min(lens)}-{max(lens)} aa):')
    print(f'  SW    mediana {statistics.median(sw_ms):8.1f} ms  '
          f'rango {min(sw_ms):.1f} - {max(sw_ms):.1f}')
    print(f'  DEDAL mediana {statistics.median(dd_ms):8.1f} ms  '
          f'rango {min(dd_ms):.1f} - {max(dd_ms):.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(HERE, 'results_tuned.csv'))
    args = parser.parse_args()

    rows = load(args.csv)
    summary(rows)
    plot_f1_vs_pid(rows, os.path.join(HERE, 'f1_vs_pid.png'))
    plot_time_vs_length(rows, os.path.join(HERE, 'tiempo_vs_longitud.png'))


if __name__ == '__main__':
    main()
