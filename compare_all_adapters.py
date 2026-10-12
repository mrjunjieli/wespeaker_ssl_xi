#!/usr/bin/env python3
"""Compare all adapter variants with different bottleneck dimensions."""

import torch
from wespeaker.models.ssl_backend import (
    SSL_BACKEND_MHFA_LN_Gaussian,
    SSL_BACKEND_MHFA_LN_Gaussian_Adapter,
    SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter,
)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_variant(model_class, name, adapter_dim=None):
    """Test a model variant and return parameter count."""
    kwargs = {
        'head_nb': 8,
        'feat_dim': 768,
        'compression_dim': 128,
        'embed_dim': 256,
        'nb_layer': 13,
    }

    if adapter_dim is not None:
        kwargs['adapter_dim'] = adapter_dim

    model = model_class(**kwargs)
    params = count_parameters(model)

    # Test forward pass
    batch_size = 2
    frames = 50
    x = torch.randn(batch_size, 768, frames, 13)

    model.eval()
    with torch.no_grad():
        value, embedding_variance, embedding = model(x)

    return params, embedding.shape


def main():
    print("=" * 80)
    print("PARAMETER COMPARISON: MHFA_LN_Gaussian Variants")
    print("=" * 80)

    results = []

    # Original model
    params, shape = test_model_variant(SSL_BACKEND_MHFA_LN_Gaussian, "Original (no adapter)")
    results.append(("Original (no adapter)", None, params, 0, shape))
    baseline = params

    # Test different adapter dimensions
    adapter_configs = [
        ("LightAdapter", 64),
        ("LightAdapter", 96),
        ("LightAdapter", 128),
        ("LightAdapter", 192),
        ("Adapter (original)", 256),
    ]

    for config_name, adapter_dim in adapter_configs:
        if adapter_dim == 256:
            model_class = SSL_BACKEND_MHFA_LN_Gaussian_Adapter
        else:
            model_class = SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter

        params, shape = test_model_variant(model_class, config_name, adapter_dim)
        increase = params - baseline
        percentage = (params / baseline - 1) * 100
        results.append((config_name, adapter_dim, params, increase, shape))

    # Print results table
    print(f"\n{'Model':<30} {'Adapter Dim':<12} {'Total Params':<15} {'Added':<15} {'% Increase':<12} {'Output Shape'}")
    print("-" * 110)

    for name, adapter_dim, params, increase, shape in results:
        adapter_str = "—" if adapter_dim is None else str(adapter_dim)
        increase_str = "—" if increase == 0 else f"{increase:,}"
        pct_str = "—" if increase == 0 else f"+{(params/baseline - 1)*100:.1f}%"
        print(f"{name:<30} {adapter_str:<12} {params:>14,} {increase_str:>14} {pct_str:<12} {str(shape)}")

    # Detailed breakdown for recommended sizes
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN - Recommended Configurations")
    print("=" * 80)

    recommended = [
        ("Original", None, SSL_BACKEND_MHFA_LN_Gaussian),
        ("LightAdapter (64)", 64, SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter),
        ("LightAdapter (128)", 128, SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter),
    ]

    for name, adapter_dim, model_class in recommended:
        print(f"\n{name}")
        print("-" * 80)

        kwargs = {
            'head_nb': 8,
            'feat_dim': 768,
            'compression_dim': 128,
            'embed_dim': 256,
            'nb_layer': 13,
            'verbose': True
        }
        if adapter_dim is not None:
            kwargs['adapter_dim'] = adapter_dim

        model = model_class(**kwargs)

        # Calculate adapter-specific params
        if adapter_dim is not None:
            adapter_params = sum(p.numel() for p in model.adapter_layers.parameters())
            total_params = count_parameters(model)

            # Per-layer breakdown
            linear1 = 768 * adapter_dim + adapter_dim
            layernorm = adapter_dim * 2
            linear2 = adapter_dim * adapter_dim + adapter_dim
            per_layer = linear1 + layernorm + linear2

            print(f"  Adapter params per layer: {per_layer:,}")
            print(f"  Total adapter params (×13): {adapter_params:,}")
            print(f"  Rest of model: {total_params - adapter_params:,}")
            print(f"  Total: {total_params:,}")
            print(f"  Increase over baseline: +{(total_params/baseline - 1)*100:.1f}%")


if __name__ == "__main__":
    main()
