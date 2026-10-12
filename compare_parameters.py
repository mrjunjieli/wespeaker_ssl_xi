#!/usr/bin/env python3
"""Compare parameters between original and adapter models."""

import torch
from wespeaker.models.ssl_backend import (
    SSL_BACKEND_MHFA_LN_Gaussian,
    SSL_BACKEND_MHFA_LN_Gaussian_Adapter
)


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def analyze_parameter_distribution(model, model_name):
    """Analyze parameter distribution by component."""
    print(f"\n{'='*80}")
    print(f"{model_name} - Parameter Breakdown")
    print(f"{'='*80}")

    total_params = 0
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"{name:30s}: {params:>12,} parameters")

        # If it's a ModuleList, show per-item details
        if isinstance(module, torch.nn.ModuleList):
            if len(module) > 0:
                per_item = sum(p.numel() for p in module[0].parameters())
                print(f"  {'(per item)':28s}: {per_item:>12,} × {len(module)} items")

    print(f"{'-'*80}")
    print(f"{'TOTAL':30s}: {total_params:>12,} parameters")
    return total_params


def main():
    # Original model
    original = SSL_BACKEND_MHFA_LN_Gaussian(
        head_nb=8,
        feat_dim=768,
        compression_dim=128,
        embed_dim=256,
        nb_layer=13,
    )

    orig_total, orig_trainable = count_parameters(original)
    analyze_parameter_distribution(original, "Original MHFA_LN_Gaussian")

    # Adapter model
    adapter = SSL_BACKEND_MHFA_LN_Gaussian_Adapter(
        head_nb=8,
        feat_dim=768,
        adapter_dim=256,
        compression_dim=128,
        embed_dim=256,
        nb_layer=13,
    )

    adapt_total, adapt_trainable = count_parameters(adapter)
    analyze_parameter_distribution(adapter, "MHFA_LN_Gaussian_Adapter")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"Original model:        {orig_total:>12,} parameters")
    print(f"Adapter model:         {adapt_total:>12,} parameters")
    print(f"{'Added:':20s} {adapt_total - orig_total:>12,} parameters ({(adapt_total/orig_total - 1)*100:.1f}% increase)")

    # Calculate adapter layer contribution
    adapter_params = sum(p.numel() for p in adapter.adapter_layers.parameters())
    print(f"\nAdapter layers alone:  {adapter_params:>12,} parameters ({adapter_params/adapt_total*100:.1f}% of total)")

    # Breakdown of adapter computation
    print(f"\n{'='*80}")
    print("ADAPTER LAYER BREAKDOWN (per layer)")
    print(f"{'='*80}")
    feat_dim = 768
    adapter_dim = 256

    linear1 = feat_dim * adapter_dim + adapter_dim  # weights + bias
    layernorm = adapter_dim * 2  # weight + bias
    linear2 = adapter_dim * adapter_dim + adapter_dim  # weights + bias

    per_layer = linear1 + layernorm + linear2

    print(f"Linear(768→256):       {linear1:>12,} parameters")
    print(f"LayerNorm(256):        {layernorm:>12,} parameters")
    print(f"Linear(256→256):       {linear2:>12,} parameters")
    print(f"{'-'*80}")
    print(f"Per adapter layer:     {per_layer:>12,} parameters")
    print(f"Total (×13 layers):    {per_layer * 13:>12,} parameters")


if __name__ == "__main__":
    main()
