#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#
# Plot the uncertainty-multiplier trajectory logged during training by
# wespeaker/utils/executor.py (uncertainty_scale_log.csv), producing a
# TensorBoard-style curve so the per-epoch / per-step scale change is visible.
#
# Usage:
#   python local/plot_uncertainty_scale.py \
#       --csv exp/.../uncertainty_scale_log.csv \
#       --out exp/.../uncertainty_scale.png \
#       --x global_step        # or: epoch
#
# If --out is omitted, the PNG is written next to the CSV.

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def load_csv(path):
    rows = []
    with open(path, newline='') as fin:
        for r in csv.DictReader(fin):
            rows.append({k: (float(v) if v not in ('', None) else float('nan'))
                         for k, v in r.items()})
    if not rows:
        raise RuntimeError(f'No rows in {path}')
    return rows


def col(rows, name):
    return [r[name] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='uncertainty_scale_log.csv')
    ap.add_argument('--out', default=None, help='output PNG path')
    ap.add_argument('--x', default='global_step',
                    choices=['global_step', 'epoch', 'iter'],
                    help='x-axis field')
    args = ap.parse_args()

    rows = load_csv(args.csv)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                   'uncertainty_scale.png')
    x = col(rows, args.x)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    # (1) multiplier: mean with [p10, p90] band and min/max envelope.
    ax = axes[0]
    ax.plot(x, col(rows, 'mult_mean'), color='C0', lw=1.5, label='mean')
    ax.fill_between(x, col(rows, 'mult_p10'), col(rows, 'mult_p90'),
                    color='C0', alpha=0.25, label='p10-p90')
    ax.plot(x, col(rows, 'mult_min'), color='C0', lw=0.6, alpha=0.5, ls='--',
            label='min/max')
    ax.plot(x, col(rows, 'mult_max'), color='C0', lw=0.6, alpha=0.5, ls='--')
    ax.axhline(1.0, color='grey', lw=0.8, ls=':')
    ax.set_ylabel('uncertainty_multiplier')
    ax.set_title('Uncertainty multiplier over training')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) effective scale = mean multiplier * base scale.
    ax = axes[1]
    ax.plot(x, col(rows, 'eff_scale_mean'), color='C1', lw=1.5)
    ax.set_ylabel('eff_scale_mean\n(mult_mean * scale)')
    ax.grid(True, alpha=0.3)

    # (3) uncertainty_weight schedule for context.
    ax = axes[2]
    ax.plot(x, col(rows, 'uncertainty_weight'), color='C2', lw=1.5)
    ax.set_ylabel('uncertainty_weight')
    ax.set_xlabel(args.x)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'Saved plot to {out}')


if __name__ == '__main__':
    main()
