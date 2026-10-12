#!/usr/bin/env python3
"""
Inspect effective ArcMargin scale distribution across mhfa_gaussian checkpoints.

Loads each checkpoint, runs synthetic SSL-format features through the
SSL_BACKEND_MHFA_LN_Gaussian backbone and ArcMarginProduct_uncertainty_inter_intra
projection, and reports the per-sample uncertainty_multiplier and effective scale
(= projection.scale * uncertainty_multiplier) without updating any parameters.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from wespeaker.models.ssl_backend.MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_Gaussian
from wespeaker.models.projections import get_projection
from wespeaker.utils.checkpoint import load_checkpoint


def compute_uncertainty_stats(projection, embeddings, covariances, labels):
    """Re-implement the uncertainty calculation from projections.py to capture stats."""
    weight = projection.weight
    cosine = F.linear(F.normalize(embeddings), F.normalize(weight))
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    target_cos = cosine.gather(1, labels.view(-1, 1))
    target_mask = F.one_hot(labels, num_classes=cosine.shape[1]).bool()
    max_non_target = cosine.masked_fill(
        target_mask, float('-inf')).max(dim=1, keepdim=True).values

    cosine_gap = target_cos - max_non_target    # inter: how far target is above nearest non-target
    inter = cosine_gap
    intra = torch.exp(cosine.max(dim=1, keepdim=True).values)

    epsilon = 1e-6
    gap_offset = -cosine_gap + 0.5
    covariance_denominator = (covariances + gap_offset + epsilon).clamp_min(epsilon)
    numerator = embeddings.square().sum(dim=1, keepdim=True)
    denominator = (embeddings.square() * covariance_denominator).sum(
        dim=1, keepdim=True).clamp_min(epsilon)

    uncertainty_scale = torch.sqrt(numerator / denominator)
    full_multiplier = uncertainty_scale * torch.exp(inter * intra)
    multiplier = 1.0 + projection.uncertainty_weight * (full_multiplier - 1.0)
    effective_scale = multiplier * projection.scale

    return {
        'uncertainty_scale': uncertainty_scale.squeeze(1).cpu().numpy(),
        'multiplier': multiplier.squeeze(1).cpu().numpy(),
        'effective_scale': effective_scale.squeeze(1).cpu().numpy(),
        'cosine_gap': cosine_gap.squeeze(1).cpu().numpy(),
        'inter_times_intra': (inter * intra).squeeze(1).cpu().numpy(),
    }


def stats_line(arr):
    return (f"min={arr.min():.3f}  p5={np.percentile(arr,5):.3f}  "
            f"p25={np.percentile(arr,25):.3f}  median={np.median(arr):.3f}  "
            f"p75={np.percentile(arr,75):.3f}  p95={np.percentile(arr,95):.3f}  "
            f"max={arr.max():.3f}  mean={arr.mean():.3f}  std={arr.std():.3f}")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base, 'exp/v2_w2vbert_sv/mhfa_gaussian/config.yaml')
    model_dir = os.path.join(base, 'exp/v2_w2vbert_sv/mhfa_gaussian/models')

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    model_args = configs['model_args']
    proj_args = configs['projection_args'].copy()
    # num_class in config may not match checkpoint (speed_perturb expansion)
    # — detect it from the first available checkpoint instead
    for epoch in [5, 10, 15, 20]:
        probe_path = os.path.join(model_dir, f'model_{epoch}.pt')
        if os.path.exists(probe_path):
            ckpt_probe = torch.load(probe_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt_probe, dict) and 'state_dict' in ckpt_probe:
                ckpt_probe = ckpt_probe['state_dict']
            if 'projection.weight' in ckpt_probe:
                proj_args['num_class'] = ckpt_probe['projection.weight'].shape[0]
            break

    num_class = proj_args['num_class']
    device = torch.device('cuda:0')  # GPU index 0 after CUDA_VISIBLE_DEVICES=2,3

    epochs = [5, 10, 15, 20]
    # synthetic input: [B, feat_dim, T, nb_layer]
    B = 64
    T = 150
    feat_dim = model_args['feat_dim']       # 1024
    nb_layer = model_args['nb_layer']       # 25
    n_batches = 20                          # 20 x 64 = 1280 samples per checkpoint

    print(f"Config: {config_path}")
    print(f"num_class={num_class}  B={B}  T={T}  n_batches={n_batches}")
    print(f"Nominal projection scale: {proj_args['scale']}")
    print()

    for epoch in epochs:
        ckpt = os.path.join(model_dir, f'model_{epoch}.pt')
        if not os.path.exists(ckpt):
            print(f"model_{epoch}.pt not found, skipping")
            continue

        model = SSL_BACKEND_MHFA_LN_Gaussian(**model_args)
        projection = get_projection(proj_args)
        projection.uncertainty_weight = 1.0  # inference-time setting

        # Add projection so load_checkpoint picks up "projection.*" keys
        model.add_module('projection', projection)
        load_checkpoint(model, ckpt)
        model = model.to(device)
        projection = model.projection
        projection.uncertainty_weight = 1.0

        model.eval()

        all_stats = {k: [] for k in [
            'uncertainty_scale', 'multiplier', 'effective_scale',
            'cosine_gap', 'inter_times_intra']}

        with torch.no_grad():
            for _ in range(n_batches):
                x = torch.randn(B, feat_dim, T, nb_layer, device=device)
                labels = torch.randint(0, num_class, (B,), device=device)

                outputs = model(x)        # (value, embedding_variance, embedding)
                embedding = outputs[-1]   # [B, embed_dim]
                covariance = outputs[1]   # [B, embed_dim]

                batch_stats = compute_uncertainty_stats(
                    projection, embedding, covariance, labels)
                for k, v in batch_stats.items():
                    all_stats[k].append(v)

        print(f"=== model_{epoch}.pt  (uncertainty_weight={projection.uncertainty_weight:.1f}) ===")
        for k, chunks in all_stats.items():
            arr = np.concatenate(chunks)
            print(f"  {k:<26}  {stats_line(arr)}")
        print()


if __name__ == '__main__':
    main()
