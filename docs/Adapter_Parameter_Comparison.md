# Parameter Comparison: MHFA_LN_Gaussian Adapter Variants

## Summary Table

| Model | Adapter Dim | Total Params | Added Params | % Increase |
|-------|-------------|--------------|--------------|------------|
| **Original (no adapter)** | — | **463,914** | — | — |
| **LightAdapter (recommended)** | 64 | **979,242** | 515,328 | **+111%** |
| LightAdapter | 96 | 1,375,146 | 911,232 | +196% |
| **LightAdapter** | 128 | **1,797,674** | 1,333,760 | **+288%** |
| LightAdapter | 192 | 2,722,602 | 2,258,688 | +487% |
| Adapter (original) | 256 | 3,754,026 | 3,290,112 | +709% |

## Recommendations

### 1. **LightAdapter with adapter_dim=64** (Conservative)
- **Total params**: 979K (+111%)
- **Added params**: 515K
- **Use when**: You want minimal parameter overhead while still having per-layer transformation
- **Per-layer adapter**: 53,504 params × 13 layers = 695K
- **Trade-off**: Smallest bottleneck, may limit representation capacity

### 2. **LightAdapter with adapter_dim=128** (Balanced) ⭐ Recommended
- **Total params**: 1.8M (+288%)
- **Added params**: 1.3M  
- **Use when**: You want a good balance between capacity and parameter efficiency
- **Per-layer adapter**: 115,200 params × 13 layers = 1.5M
- **Trade-off**: 2x larger bottleneck than 64, but still ~2x smaller than the original 256-dim adapter

### 3. **Adapter with adapter_dim=256** (Maximum capacity)
- **Total params**: 3.8M (+709%)
- **Added params**: 3.3M
- **Use when**: Parameter count is not a concern and you want maximum transformation capacity
- **Per-layer adapter**: 263,168 params × 13 layers = 3.4M
- **Trade-off**: Very large parameter increase

## Parameter Breakdown per Adapter Layer

| Adapter Dim | Linear1 (768→D) | LayerNorm | Linear2 (D→D) | Per Layer | ×13 Layers |
|-------------|-----------------|-----------|---------------|-----------|------------|
| 64 | 49,216 | 128 | 4,160 | 53,504 | 695,552 |
| 128 | 98,432 | 256 | 16,512 | 115,200 | 1,497,600 |
| 256 | 196,864 | 512 | 65,792 | 263,168 | 3,421,184 |

## Architecture Visualization

```
Original MHFA_LN_Gaussian:
SSL [768, T, 13] → Layer Fusion → K/V Compress [128] → Pooling → Embed [256]
Total: 464K params

With LightAdapter (adapter_dim=128):
SSL [768, T, 13] 
  → Per-layer: Linear(768→128) → LN → ReLU → Linear(128→128)
  → [128, T, 13]
  → Layer Fusion 
  → K/V Compress [128] 
  → Pooling 
  → Embed [256]
Total: 1.8M params (+1.3M from adapters)
```

## Usage Examples

### Configuration File (YAML)

```yaml
# Conservative (64-dim adapter)
model_args:
  ssl_backend: SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter
  ssl_backend_args:
    adapter_dim: 64
    head_nb: 8
    feat_dim: 768
    compression_dim: 128
    embed_dim: 256
    nb_layer: 13

# Balanced (128-dim adapter) - Recommended
model_args:
  ssl_backend: SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter
  ssl_backend_args:
    adapter_dim: 128
    head_nb: 8
    feat_dim: 768
    compression_dim: 128
    embed_dim: 256
    nb_layer: 13
```

### Python Code

```python
from wespeaker.models.ssl_backend import SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter

# Balanced configuration (recommended)
model = SSL_BACKEND_MHFA_LN_Gaussian_LightAdapter(
    head_nb=8,
    feat_dim=768,
    adapter_dim=128,  # Adjust: 64, 96, 128, 192, or 256
    compression_dim=128,
    embed_dim=256,
    nb_layer=13,
    verbose=True
)
```

## Design Rationale

### Why Per-Layer Adapters?
Each of the 13 SSL transformer layers captures different levels of abstraction:
- Early layers: Low-level acoustic features
- Middle layers: Phonetic representations
- Late layers: High-level semantic content

Independent adapters allow each layer to learn its own optimal transformation.

### Why Smaller Bottleneck?
- **Original (256-dim)**: Same size as embedding output, may be redundant
- **LightAdapter (64-128)**: Forces compact, discriminative representations
- **Parameter efficiency**: 64-dim uses only ~700K params vs 3.4M for 256-dim
- **Regularization effect**: Smaller bottleneck may prevent overfitting

## Comparison with w2v-BERT-2.0_SV

The original w2v-BERT-2.0_SV uses adapter_dim=128 as default, which aligns with our balanced recommendation. Their design concatenates adapted layers before pooling, while ours fuses them with learnable weights.

## Next Steps

1. **Start with adapter_dim=128** for initial experiments
2. **Try adapter_dim=64** if memory/parameters are constrained
3. **Experiment with adapter_dim=192 or 256** if baseline performance is insufficient
4. **Monitor**: Compare EER on validation set across different adapter dimensions

## Files Created

- `MHFA_LN_Gaussian_Adapter.py` - Original 256-dim adapter
- `MHFA_LN_Gaussian_LightAdapter.py` - Configurable lightweight adapter (64-256)
- `compare_all_adapters.py` - Parameter comparison script
- `test_adapter_model.py` - Basic functionality test
