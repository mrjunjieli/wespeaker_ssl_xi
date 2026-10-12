#!/usr/bin/env python3
"""Test script for MHFA_LN_Gaussian_Adapter model."""

import torch
from wespeaker.models.ssl_backend import (
    SSL_BACKEND_MHFA_LN_Gaussian_Adapter,
    SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter
)


def test_adapter_model():
    """Test forward pass of adapter models."""
    print("=" * 80)
    print("Testing SSL_BACKEND_MHFA_LN_Gaussian_Adapter")
    print("=" * 80)

    # Model parameters
    batch_size = 4
    feat_dim = 768
    frames = 100
    nb_layer = 13

    # Create model
    model = SSL_BACKEND_MHFA_LN_Gaussian_Adapter(
        head_nb=8,
        feat_dim=feat_dim,
        adapter_dim=256,
        compression_dim=128,
        embed_dim=256,
        nb_layer=nb_layer,
        feature_grad_mult=1.0,
        train_prior_mean=True,
        train_prior_prec=True,
        verbose=True
    )

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dummy input
    x = torch.randn(batch_size, feat_dim, frames, nb_layer)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        value, embedding_variance, embedding = model(x)

    print(f"\nOutput shapes:")
    print(f"  value: {value.shape}")
    print(f"  embedding_variance: {embedding_variance.shape}")
    print(f"  embedding: {embedding.shape}")

    # Check output values
    print(f"\nOutput statistics:")
    print(f"  embedding mean: {embedding.mean().item():.6f}")
    print(f"  embedding std: {embedding.std().item():.6f}")
    print(f"  variance mean: {embedding_variance.mean().item():.6f}")
    print(f"  variance std: {embedding_variance.std().item():.6f}")

    print("\n✓ SSL_BACKEND_MHFA_LN_Gaussian_Adapter test passed!")

    # Test LogPrec variant
    print("\n" + "=" * 80)
    print("Testing SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter")
    print("=" * 80)

    model_logprec = SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter(
        head_nb=8,
        feat_dim=feat_dim,
        adapter_dim=256,
        compression_dim=128,
        embed_dim=256,
        nb_layer=nb_layer,
        feature_grad_mult=1.0,
        train_prior_mean=True,
        train_prior_prec=True,
        verbose=True
    )

    model_logprec.eval()
    with torch.no_grad():
        value_lp, embedding_variance_lp, embedding_lp = model_logprec(x)

    print(f"\nOutput shapes:")
    print(f"  value: {value_lp.shape}")
    print(f"  embedding_variance: {embedding_variance_lp.shape}")
    print(f"  embedding: {embedding_lp.shape}")

    print("\n✓ SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter test passed!")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_adapter_model()
