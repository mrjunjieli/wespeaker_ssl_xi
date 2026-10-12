#!/usr/bin/env python3
"""Show detailed trainable parameters for LightAdapter models."""

import torch
from wespeaker.models.ssl_backend import (
    SSL_BACKEND_MHFA_LN_Gaussian,
    SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_LightAdapter,
)


def analyze_trainable_parameters(model, model_name):
    """Analyze trainable vs non-trainable parameters."""
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}\n")

    total_params = 0
    trainable_params = 0

    print(f"{'Component':<40} {'Total':<15} {'Trainable':<15} {'Frozen':<15}")
    print("-" * 85)

    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_frozen = module_total - module_trainable

        total_params += module_total
        trainable_params += module_trainable

        print(f"{name:<40} {module_total:>14,} {module_trainable:>14,} {module_frozen:>14,}")

        # Show details for ModuleList (adapter layers)
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            per_item = sum(p.numel() for p in module[0].parameters())
            per_item_trainable = sum(p.numel() for p in module[0].parameters() if p.requires_grad)
            print(f"  {'(per adapter layer)' :<38} {per_item:>14,} {per_item_trainable:>14,} {per_item - per_item_trainable:>14,}")

    # Show parameters that are direct children (like weights_k, prior_mean, etc.)
    direct_params = sum(p.numel() for p in model.parameters(recurse=False))
    direct_trainable = sum(p.numel() for p in model.parameters(recurse=False) if p.requires_grad)
    if direct_params > 0:
        print(f"{'(direct model parameters)':<40} {direct_params:>14,} {direct_trainable:>14,} {direct_params - direct_trainable:>14,}")
        total_params += direct_params
        trainable_params += direct_trainable

    print("-" * 85)
    print(f"{'TOTAL':<40} {total_params:>14,} {trainable_params:>14,} {total_params - trainable_params:>14,}")

    # Show percentage
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
    frozen_pct = 100 - trainable_pct
    print(f"{'Percentage':<40} {'100.0%':>14} {f'{trainable_pct:.1f}%':>14} {f'{frozen_pct:.1f}%':>14}")

    return total_params, trainable_params


def main():
    print("="*80)
    print("TRAINABLE PARAMETERS ANALYSIS")
    print("="*80)

    configs = [
        ("Original MHFA_LN_Gaussian", None, SSL_BACKEND_MHFA_LN_Gaussian),
        ("LightAdapter (adapter_dim=64)", 64, SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter),
        ("LightAdapter (adapter_dim=128)", 128, SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter),
        ("LightAdapter (adapter_dim=256)", 256, SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter),
        ("LogPrec LightAdapter (adapter_dim=128)", 128, SSL_BACKEND_MHFA_LN_GaussianLogPrec_LightAdapter),
    ]

    results = []

    for name, adapter_dim, model_class in configs:
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
        total, trainable = analyze_trainable_parameters(model, name)
        results.append((name, adapter_dim, total, trainable))

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}\n")
    print(f"{'Model':<45} {'Adapter Dim':<12} {'Total':<15} {'Trainable':<15}")
    print("-" * 87)

    for name, adapter_dim, total, trainable in results:
        adapter_str = "—" if adapter_dim is None else str(adapter_dim)
        print(f"{name:<45} {adapter_str:<12} {total:>14,} {trainable:>14,}")

    print("\n" + "="*80)
    print("NOTE: All parameters are trainable by default in these models.")
    print("If you freeze the SSL frontend, only backend parameters will be updated.")
    print("="*80)


if __name__ == "__main__":
    main()
