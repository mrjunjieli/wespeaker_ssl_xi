#!/usr/bin/env python3
"""
Plot histogram of scale × uncertainty_scale using REAL training data.

Step 1: Load actual audio from data/vox2_dev/shard.list and run the frozen
        W2V-BERT frontend once to extract features. Cache to CPU.
Step 2: For each checkpoint (model_5/10/15/20), run only the MHFA_LN_Gaussian
        backend + projection on the cached features and collect stats.
"""

import sys, os, itertools
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, os.path.abspath(ROOT))

from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import frontend_class_dict
from wespeaker.models.ssl_backend.MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_Gaussian
from wespeaker.models.projections import get_projection
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import spk2id


# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..')
CONFIG_PATH = os.path.join(BASE, 'exp/v2_w2vbert_sv/mhfa_gaussian/config.yaml')
MODEL_DIR   = os.path.join(BASE, 'exp/v2_w2vbert_sv/mhfa_gaussian/models')
OUT_PATH    = os.path.join(BASE, 'exp/v2_w2vbert_sv/mhfa_gaussian/scale_uncertainty_hist_real.png')

N_BATCHES   = 400    # 400 × 32 = 12800 real train utterances
BATCH_SIZE  = 32
DEVICE      = torch.device('cuda:0')   # after CUDA_VISIBLE_DEVICES=2,3

with open(CONFIG_PATH) as f:
    configs = yaml.safe_load(f)

model_args = configs['model_args']
proj_args  = configs['projection_args'].copy()

# Detect num_class from first available checkpoint
for ep in [5, 10, 15, 20]:
    p = os.path.join(MODEL_DIR, f'model_{ep}.pt')
    if os.path.exists(p):
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if 'projection.weight' in ckpt:
            proj_args['num_class'] = ckpt['projection.weight'].shape[0]
        break
num_class = proj_args['num_class']
print(f"num_class from checkpoint: {num_class}  |  nominal scale: {proj_args['scale']}")


# ── Step 1: build dataset + dataloader ───────────────────────────────────────
print("\n[1/3] Building dataset from real training shards...")

train_utt_spk_list = read_table(configs['train_label'])
spk2id_dict = spk2id(train_utt_spk_list)

# Disable speed_perturb and augmentation for clean inference
dataset_args = dict(configs['dataset_args'])
dataset_args['speed_perturb'] = False
dataset_args['spec_aug']      = False
dataset_args['aug_prob']      = 0.0

train_dataset = Dataset(
    configs['data_type'],
    configs['train_data'],
    dataset_args,
    spk2id_dict,
    repeat_dataset=True,   # infinite; we stop via islice
    reverb_lmdb_file=None,
    noise_lmdb_file=None,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)


# ── Step 2: extract features with frozen W2V-BERT frontend ───────────────────
print(f"[2/3] Extracting features from {N_BATCHES} batches × {BATCH_SIZE} = "
      f"{N_BATCHES*BATCH_SIZE} utterances ...")

frontend_type = dataset_args.get('frontend', 'fbank')
frontend_args_key = frontend_type + '_args'
frontend = frontend_class_dict[frontend_type](
    **dataset_args[frontend_args_key],
    sample_rate=dataset_args['resample_rate'],
)
frontend = frontend.to(DEVICE)
frontend.eval()

feature_cache = []   # list of [B, feat_dim, T, nb_layer] CPU tensors
label_cache   = []   # list of [B] CPU tensors

for batch in itertools.islice(train_loader, N_BATCHES):
    wavs    = batch['wav'].squeeze(1).float().to(DEVICE)       # [B, W]
    wavs_len = torch.full((wavs.shape[0],), wavs.shape[1],
                          dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        feats, _ = frontend(wavs, wavs_len)                    # [B, D, T, L]
    feature_cache.append(feats.cpu())
    label_cache.append(batch['label'].cpu())
    print(f"  batch {len(feature_cache)}/{N_BATCHES}  feats={feats.shape}", end='\r')

print(f"\n  Done. Total samples: {sum(f.shape[0] for f in feature_cache)}")
del frontend  # free GPU memory


# ── Step 3: per-checkpoint inference ─────────────────────────────────────────
def compute_uncertainty_scale(model, projection, feats_cpu, labels_cpu):
    feats   = feats_cpu.to(DEVICE)
    labels  = labels_cpu.long().to(DEVICE)
    outputs = model(feats)          # (value, embedding_variance, embedding)
    embedding  = outputs[-1]        # [B, D]
    covariance = outputs[1]         # [B, D]

    weight = projection.weight
    cosine = F.linear(F.normalize(embedding), F.normalize(weight))
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    target_cos   = cosine.gather(1, labels.view(-1, 1))
    target_mask  = F.one_hot(labels, num_classes=cosine.shape[1]).bool()
    max_non_target = cosine.masked_fill(
        target_mask, float('-inf')).max(dim=1, keepdim=True).values

    cosine_gap  = target_cos - max_non_target
    epsilon     = 1e-6
    gap_offset  = -cosine_gap + 0.5
    cov_denom   = (covariance + gap_offset + epsilon).clamp_min(epsilon)
    numerator   = embedding.square().sum(dim=1, keepdim=True)
    denominator = (embedding.square() * cov_denom).sum(
        dim=1, keepdim=True).clamp_min(epsilon)
    unc_scale   = torch.sqrt(numerator / denominator)   # [B, 1]
    return unc_scale.squeeze(1).cpu().numpy()


print("\n[3/3] Running backend inference per checkpoint...")

epochs = [5, 10, 15, 20]
colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
nominal_scale = proj_args['scale']

results = {}
for epoch in epochs:
    ckpt_path = os.path.join(MODEL_DIR, f'model_{epoch}.pt')
    if not os.path.exists(ckpt_path):
        print(f"  model_{epoch}.pt not found, skipping")
        continue
    print(f"  Loading model_{epoch}.pt ...", end=' ', flush=True)

    backend    = SSL_BACKEND_MHFA_LN_Gaussian(**model_args)
    projection = get_projection(proj_args)
    backend.add_module('projection', projection)
    load_checkpoint(backend, ckpt_path)
    backend    = backend.to(DEVICE)
    projection = backend.projection
    backend.eval()

    all_unc = []
    with torch.no_grad():
        for feats, labels in zip(feature_cache, label_cache):
            unc = compute_uncertainty_scale(backend, projection, feats, labels)
            all_unc.append(unc)

    arr = np.concatenate(all_unc) * nominal_scale    # scale × uncertainty_scale
    results[epoch] = arr
    print(f"n={len(arr)}  mean={arr.mean():.2f}  std={arr.std():.2f}  "
          f"min={arr.min():.2f}  max={arr.max():.2f}")
    del backend


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (epoch, color) in enumerate(zip(epochs, colors)):
    if epoch not in results:
        continue
    arr = results[epoch]
    ax  = axes[idx]
    ax.hist(arr, bins=80, color=color, edgecolor='none', alpha=0.85)
    ax.axvline(arr.mean(), color='#111', linewidth=1.5, linestyle='--',
               label=f'mean={arr.mean():.2f}')
    ax.axvline(nominal_scale, color='#e15759', linewidth=1.2, linestyle=':',
               label=f'nominal={nominal_scale:.0f}')
    ax.set_title(f'model_{epoch}  (n={len(arr)})', fontsize=13, fontweight='bold')
    ax.set_xlabel('scale × uncertainty_scale', fontsize=11)
    ax.set_ylabel('count', fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(
    'Distribution of scale × uncertainty_scale\n'
    '(mhfa_gaussian, real VoxCeleb2 training utterances)',
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")
