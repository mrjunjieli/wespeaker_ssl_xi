#!/usr/bin/env python3
"""
Plot histogram of scale * uncertainty_scale across mhfa_gaussian checkpoints.
X-axis: scale * uncertainty_scale  (nominal scale=32, so this is 32 * uncertainty_scale)
Y-axis: count
"""

import sys
import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from wespeaker.models.ssl_backend.MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_Gaussian
from wespeaker.models.projections import get_projection
from wespeaker.utils.checkpoint import load_checkpoint


def collect_scale_uncertainty(model_path, proj_args, model_args, device,
                               n_batches=30, B=64, T=150):
    """Return array of scale * uncertainty_scale for n_batches*B samples."""
    feat_dim = model_args['feat_dim']
    nb_layer = model_args['nb_layer']
    num_class = proj_args['num_class']
    nominal_scale = proj_args['scale']

    model = SSL_BACKEND_MHFA_LN_Gaussian(**model_args)
    projection = get_projection(proj_args)
    model.add_module('projection', projection)
    load_checkpoint(model, model_path)
    model = model.to(device)
    model.eval()

    all_uncertainty_scale = []

    with torch.no_grad():
        for _ in range(n_batches):
            x = torch.randn(B, feat_dim, T, nb_layer, device=device)
            labels = torch.randint(0, num_class, (B,), device=device)

            outputs = model(x)          # (value, variance, embedding)
            embedding = outputs[-1]     # [B, D]
            covariance = outputs[1]     # [B, D]

            weight = model.projection.weight
            cosine = F.linear(F.normalize(embedding), F.normalize(weight))
            cosine = cosine.clamp(-1e-7 + (-1.0), 1.0 - 1e-7)

            target_cos = cosine.gather(1, labels.view(-1, 1))
            target_mask = F.one_hot(labels, num_classes=cosine.shape[1]).bool()
            max_non_target = cosine.masked_fill(
                target_mask, float('-inf')).max(dim=1, keepdim=True).values

            cosine_gap = target_cos - max_non_target
            epsilon = 1e-6
            gap_offset = -cosine_gap + 0.5
            covariance_denominator = (covariance + gap_offset + epsilon).clamp_min(epsilon)
            numerator = embedding.square().sum(dim=1, keepdim=True)
            denominator = (embedding.square() * covariance_denominator).sum(
                dim=1, keepdim=True).clamp_min(epsilon)
            uncertainty_scale = torch.sqrt(numerator / denominator)  # [B, 1]

            all_uncertainty_scale.append(uncertainty_scale.squeeze(1).cpu().numpy())

    arr = np.concatenate(all_uncertainty_scale)
    return arr * nominal_scale   # scale * uncertainty_scale


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, 'exp/v2_w2vbert_sv/mhfa_gaussian/config.yaml')
    model_dir  = os.path.join(base, 'exp/v2_w2vbert_sv/mhfa_gaussian/models')
    out_path   = os.path.join(base, 'exp/v2_w2vbert_sv/mhfa_gaussian/scale_uncertainty_hist.png')

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    model_args = configs['model_args']
    proj_args  = configs['projection_args'].copy()

    # detect num_class from checkpoint
    for ep in [5, 10, 15, 20]:
        p = os.path.join(model_dir, f'model_{ep}.pt')
        if os.path.exists(p):
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            if 'projection.weight' in ckpt:
                proj_args['num_class'] = ckpt['projection.weight'].shape[0]
            break

    device = torch.device('cuda:0')
    epochs = [5, 10, 15, 20]
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    axes = axes.flatten()

    nominal_scale = proj_args['scale']   # 32.0
    print(f"Nominal scale: {nominal_scale}  |  collecting {30*64} samples per checkpoint")

    for idx, epoch in enumerate(epochs):
        ckpt = os.path.join(model_dir, f'model_{epoch}.pt')
        if not os.path.exists(ckpt):
            print(f"  model_{epoch}.pt not found, skipping")
            continue

        print(f"  Loading model_{epoch}.pt ...")
        vals = collect_scale_uncertainty(ckpt, proj_args, model_args, device)

        ax = axes[idx]
        ax.hist(vals, bins=60, color=colors[idx], edgecolor='none', alpha=0.85)
        ax.axvline(vals.mean(), color='#222', linewidth=1.4, linestyle='--', label=f'mean={vals.mean():.2f}')
        ax.axvline(nominal_scale, color='#e15759', linewidth=1.2, linestyle=':', label=f'nominal={nominal_scale:.0f}')
        ax.set_title(f'model_{epoch}', fontsize=13, fontweight='bold')
        ax.set_xlabel('scale × uncertainty_scale', fontsize=11)
        ax.set_ylabel('count', fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        print(f"    min={vals.min():.2f}  mean={vals.mean():.2f}  median={np.median(vals):.2f}  "
              f"p95={np.percentile(vals,95):.2f}  max={vals.max():.2f}  std={vals.std():.2f}")

    fig.suptitle('Distribution of scale × uncertainty_scale\n(mhfa_gaussian, training-set synthetic inference)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
