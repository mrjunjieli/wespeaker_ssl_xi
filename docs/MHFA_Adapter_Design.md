# MHFA_LN_Gaussian with Adapter Layers

## Overview

Added adapter transformation layers between SSL features and layer fusion in the MHFA_LN_Gaussian backend, following the design from [w2v-BERT-2.0_SV](https://github.com/ZXHY-82/w2v-BERT-2.0_SV).

## Architecture Changes

### Original MHFA_LN_Gaussian Flow
```
SSL Features [B, 768, T, 13]
    ↓
Layer Fusion (weighted sum)
    ↓
K/V Compression (768 → 128)
    ↓
MHFA + Gaussian Pooling
    ↓
Embedding [B, 256]
```

### New MHFA_LN_Gaussian_Adapter Flow
```
SSL Features [B, 768, T, 13]
    ↓
Per-Layer Adapters (13 independent modules)
  Linear(768 → 256) → LayerNorm → ReLU → Linear(256 → 256)
    ↓ [B, 256, T, 13]
Layer Fusion (weighted sum)
    ↓
K/V Compression (256 → 128)
    ↓
MHFA + Gaussian Pooling
    ↓
Embedding [B, 256]
```

## Implementation Details

### Adapter Module Structure
Each of the 13 SSL layers gets its own adapter:
```python
nn.Sequential(
    nn.Linear(768, 256),      # Dimension reduction
    nn.LayerNorm(256),        # Normalization
    nn.ReLU(inplace=True),    # Non-linearity
    nn.Linear(256, 256),      # Feature transformation
)
```

### Key Design Choices

1. **Per-layer independence**: Each SSL layer is transformed independently before fusion
2. **Bottleneck design**: 768 → 256 dimension reduction learns compact representations
3. **Residual-free**: No skip connection (unlike some adapter designs) to force full transformation
4. **Position**: Adapters come before gradient multiplication and layer fusion

### Parameter Count

| Model | Total Parameters |
|-------|-----------------|
| Original MHFA_LN_Gaussian | ~362K |
| MHFA_LN_Gaussian_Adapter | ~3.75M |

The adapter adds ~3.4M parameters (13 layers × 2 Linear layers × weights).

## Model Variants

Two variants are implemented:

1. **SSL_BACKEND_MHFA_LN_Gaussian_Adapter**
   - Frame precision: `softplus(attention_logits)`
   - Posterior precision: sum of frame precisions + prior precision

2. **SSL_BACKEND_MHFA_LN_GaussianLogPrec_Adapter**
   - Frame log-precision: attention logits directly
   - Posterior mean: softmax-weighted over log-precisions

## Usage

### In Configuration File

```yaml
model: w2vbert_sv
model_args:
  ssl_backend: SSL_BACKEND_MHFA_LN_Gaussian_Adapter
  ssl_backend_args:
    head_nb: 8
    feat_dim: 768
    adapter_dim: 256        # Adapter bottleneck dimension
    compression_dim: 128
    embed_dim: 256
    nb_layer: 13
    feature_grad_mult: 1.0
    train_prior_mean: True
    train_prior_prec: True
```

### Direct Instantiation

```python
from wespeaker.models.ssl_backend import SSL_BACKEND_MHFA_LN_Gaussian_Adapter

model = SSL_BACKEND_MHFA_LN_Gaussian_Adapter(
    head_nb=8,
    feat_dim=768,
    adapter_dim=256,
    compression_dim=128,
    embed_dim=256,
    nb_layer=13,
    verbose=True
)

# Input: [batch, 768, frames, 13]
value, embedding_variance, embedding = model(ssl_features)
```

## Expected Benefits

1. **Learned feature space**: Adapters can learn task-specific transformations of SSL features
2. **Layer-specific processing**: Each SSL layer gets its own transformation pathway
3. **Non-linear mapping**: ReLU allows modeling complex relationships
4. **Better fine-tuning**: Adapter parameters can be trained while keeping SSL frozen

## Testing

Run the test script:
```bash
CUDA_VISIBLE_DEVICES=2,3 python test_adapter_model.py
```

Expected output shows:
- Model structure visualization
- Parameter counts
- Output shape verification
- Basic statistics on embeddings and variances

## Files Modified/Created

- `wespeaker/models/ssl_backend/MHFA_LN_Gaussian_Adapter.py` (new)
- `wespeaker/models/ssl_backend/__init__.py` (modified)
- `test_adapter_model.py` (new test script)
