#!/usr/bin/env python3
"""
Plot histogram of the TRUE effective ArcMargin scale across checkpoints.

In ArcMarginProduct_uncertainty_inter_intra.forward() the final logits are:

    output *= uncertainty_multiplier          # 1 + w*(unc_scale*exp(inter*intra) - 1)
    return output * self.scale

so the per-sample effective scale applied to the logits is

    effective_scale = self.scale * uncertainty_multiplier
                    = 32 * unc_scale * exp(inter * intra)      (with w = 1)

This script extracts real VoxCeleb2 training features once (cached to disk),
then for each checkpoint reports the distribution of that effective scale.
"""

import sys, os, itertools

# OmegaConf's grammar_visitor calls sys.exit(1) when its antlr parsers are
# missing.  We only use it for YAML loading (no interpolation), so suppress
# that exit during imports and restore the real one immediately after.
_real_exit = sys.exit
def _exit_guard(code=0):
    if code != 0:
        pass   # swallow; we don't need the antlr parsers
    else:
        _real_exit(0)
sys.exit = _exit_guard

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, ROOT)

from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import frontend_class_dict
from wespeaker.models.ssl_backend.MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_Gaussian
from wespeaker.models.projections import get_projection
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import spk2id


BASE        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE, 'exp/v2_w2vbert_sv/mhfa_gaussian/config.yaml')
MODEL_DIR   = os.path.join(BASE, 'exp/v2_w2vbert_sv/mhfa_gaussian/models')
OUT_PATH    = os.path.join(BASE, 'exp/v2_w2vbert_sv/mhfa_gaussian/effective_scale_hist.png')
CACHE_PATH  = '/tmp/mhfa_gaussian_feats_cache.pt'

N_BATCHES  = 400
BATCH_SIZE = 32
DEVICE     = torch.device('cuda:0')

with open(CONFIG_PATH) as f:
    configs = yaml.safe_load(f)

model_args = configs['model_args']
proj_args  = configs['projection_args'].copy()

for ep in [5, 10, 15, 20]:
    p = os.path.join(MODEL_DIR, f'model_{ep}.pt')
    if os.path.exists(p):
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if 'projection.weight' in ckpt:
            proj_args['num_class'] = ckpt['projection.weight'].shape[0]
        break
num_class     = proj_args['num_class']
nominal_scale = proj_args['scale']
print(f"num_class={num_class}  nominal scale={nominal_scale}")


# ── Features: extract once, cache to disk ────────────────────────────────────
if os.path.exists(CACHE_PATH):
    print(f"\n[1/3] Loading cached features from {CACHE_PATH}")
    cache = torch.load(CACHE_PATH)
    feature_cache, label_cache = cache['feats'], cache['labels']
else:
    print("\n[1/3] Extracting features from real training shards ...")
    train_utt_spk_list = read_table(configs['train_label'])
    spk2id_dict = spk2id(train_utt_spk_list)

    dataset_args = dict(configs['dataset_args'])
    dataset_args['speed_perturb'] = False
    dataset_args['spec_aug']      = False
    dataset_args['aug_prob']      = 0.0

    ds = Dataset(configs['data_type'], configs['train_data'], dataset_args,
                 spk2id_dict, repeat_dataset=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4,
                        pin_memory=True, drop_last=True)

    frontend_type = dataset_args.get('frontend', 'fbank')
    frontend = frontend_class_dict[frontend_type](
        **dataset_args[frontend_type + '_args'],
        sample_rate=dataset_args['resample_rate'])
    frontend = frontend.to(DEVICE).eval()

    feature_cache, label_cache = [], []
    for batch in itertools.islice(loader, N_BATCHES):
        wavs = batch['wav'].squeeze(1).float().to(DEVICE)
        wavs_len = torch.full((wavs.shape[0],), wavs.shape[1],
                              dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            feats, _ = frontend(wavs, wavs_len)
        feature_cache.append(feats.cpu())
        label_cache.append(batch['label'].cpu())
        print(f"  batch {len(feature_cache)}/{N_BATCHES}", end='\r')
    total = sum(f.shape[0] for f in feature_cache)
    print(f"\n  Extracted {total} utterances; caching to {CACHE_PATH}")
    torch.save({'feats': feature_cache, 'labels': label_cache}, CACHE_PATH)
    del frontend


# ── Per-checkpoint: true effective scale from the real projection forward ────
def effective_scale_stats(model, feats_cpu, labels_cpu, w=1.0):
    """Replicates ArcMarginProduct_uncertainty_inter_intra.forward() math
    and returns the per-sample effective scale = scale * multiplier."""
    feats  = feats_cpu.to(DEVICE)
    labels = labels_cpu.long().to(DEVICE)

    with torch.no_grad():
        outputs = model(feats)
        embedding  = outputs[-1]
        covariance = outputs[1]

        weight = model.projection.weight
        cosine = F.linear(F.normalize(embedding), F.normalize(weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        target_cos = cosine.gather(1, labels.view(-1, 1))
        target_mask = F.one_hot(labels, num_classes=cosine.shape[1]).bool()
        max_non_target = cosine.masked_fill(
            target_mask, float('-inf')).max(dim=1, keepdim=True).values

        cosine_gap = target_cos - max_non_target      # inter
        inter = cosine_gap
        intra = torch.exp(cosine.max(dim=1, keepdim=True).values)

        eps = 1e-6
        gap_offset = -cosine_gap + 0.5
        cov_denom = (covariance + gap_offset + eps).clamp_min(eps)
        numerator = embedding.square().sum(dim=1, keepdim=True)
        denominator = (embedding.square() * cov_denom).sum(
            dim=1, keepdim=True).clamp_min(eps)
        unc_scale = torch.sqrt(numerator / denominator)

        full_multiplier = unc_scale * torch.exp(inter * intra)
        multiplier = 1.0 + w * (full_multiplier - 1.0)
        eff_scale = multiplier * nominal_scale

        # covariance is [B, embed_dim]; take per-sample mean for a scalar summary
        cov_mean = covariance.mean(dim=1, keepdim=True)

        return {
            'cosine_gap':     cosine_gap.squeeze(1).cpu().numpy(),
            'inter':          inter.squeeze(1).cpu().numpy(),          # == cosine_gap
            'intra':          intra.squeeze(1).cpu().numpy(),          # exp(max_cosine)
            'Lambda':         gap_offset.squeeze(1).cpu().numpy(),     # -cosine_gap + 0.5
            'covariance':     cov_mean.squeeze(1).cpu().numpy(),       # per-sample mean cov
            'unc_scale':      unc_scale.squeeze(1).cpu().numpy(),
            'effective_scale': eff_scale.squeeze(1).cpu().numpy(),
            'multiplier':     multiplier.squeeze(1).cpu().numpy(),
            'exp_gap':        torch.exp(inter * intra).squeeze(1).cpu().numpy(),
        }


print("\n[2/3] Running real projection forward per checkpoint ...")
epochs = [5, 10, 15, 20]
colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
results = {}

for epoch in epochs:
    ckpt_path = os.path.join(MODEL_DIR, f'model_{epoch}.pt')
    if not os.path.exists(ckpt_path):
        continue
    backend = SSL_BACKEND_MHFA_LN_Gaussian(**model_args)
    projection = get_projection(proj_args)
    backend.add_module('projection', projection)
    load_checkpoint(backend, ckpt_path)
    backend = backend.to(DEVICE).eval()

    chunks = {k: [] for k in [
        'cosine_gap', 'inter', 'intra', 'Lambda', 'covariance',
        'unc_scale', 'effective_scale', 'multiplier', 'exp_gap']}
    for feats, labels in zip(feature_cache, label_cache):
        st = effective_scale_stats(backend, feats, labels, w=1.0)
        for k, v in st.items():
            chunks[k].append(v)

    res = {k: np.concatenate(v) for k, v in chunks.items()}
    results[epoch] = res
    print(f"  model_{epoch}: eff_scale mean={res['effective_scale'].mean():6.2f} "
          f"(vs nominal {nominal_scale})  std={res['effective_scale'].std():.2f}  "
          f"| unc_scale mean={res['unc_scale'].mean():.3f}  "
          f"exp(inter*intra) mean={res['exp_gap'].mean():.3f}")
    del backend


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (epoch, color) in enumerate(zip(epochs, colors)):
    if epoch not in results:
        continue
    arr = results[epoch]['effective_scale']
    ax = axes[idx]
    ax.hist(arr, bins=80, color=color, edgecolor='none', alpha=0.85)
    ax.axvline(arr.mean(), color='#111', linewidth=1.5, linestyle='--',
               label=f'mean={arr.mean():.2f}')
    ax.axvline(nominal_scale, color='#e15759', linewidth=1.4, linestyle=':',
               label=f'nominal={nominal_scale:.0f}')
    ax.set_title(f'model_{epoch}  (n={len(arr)})', fontsize=13, fontweight='bold')
    ax.set_xlabel('effective scale = scale × uncertainty_multiplier', fontsize=10)
    ax.set_ylabel('count', fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(
    'True effective ArcMargin scale per checkpoint\n'
    '(mhfa_gaussian, real VoxCeleb2 training utterances, w=1)',
    fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PATH}")


# ── Detailed summary table ───────────────────────────────────────────────────
VARS = [
    ('cosine_gap',     'cosine_gap   (inter)'),
    ('Lambda',         'Lambda       (-gap+0.5)'),
    ('covariance',     'covariance   (mean/dim)'),
    ('intra',          'intra        (exp(max_cos))'),
    ('unc_scale',      'unc_scale'),
    ('effective_scale','effective_scale'),
]

HDR = f"{'':>28}{'min':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p95':>9}{'max':>9}{'mean':>9}{'std':>9}"

for ep in epochs:
    if ep not in results:
        continue
    print(f"\n{'='*90}")
    print(f"  model_{ep}   (nominal scale={nominal_scale})")
    print(HDR)
    for key, label in VARS:
        a = results[ep][key]
        print(f"  {label:<26}"
              f"{a.min():>9.4f}{np.percentile(a,25):>9.4f}{np.median(a):>9.4f}"
              f"{np.percentile(a,75):>9.4f}{np.percentile(a,95):>9.4f}"
              f"{a.max():>9.4f}{a.mean():>9.4f}{a.std():>9.4f}")
