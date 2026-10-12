#!/usr/bin/env python3
"""
Full vox2_dev extraction with model_20: saves embedding + variance arks
and computes effective-scale distribution in a single forward pass.

Usage (called from the nj-split wrapper):
  python local/extract_vox2_scale.py \
      --config  exp/v2_w2vbert_sv/mhfa_gaussian/config.yaml \
      --model_path exp/v2_w2vbert_sv/mhfa_gaussian/models/model_20.pt \
      --data_list <sub_shard_list> \
      --embed_ark <output>.ark \
      --utt2spk   data/vox2_dev/utt2spk
"""

import sys, os

# Suppress OmegaConf antlr sys.exit during import
_real_exit = sys.exit
def _exit_guard(code=0):
    if code != 0:
        pass
    else:
        _real_exit(0)
sys.exit = _exit_guard

import copy
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import kaldiio
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.exit = _real_exit   # restore after imports

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, ROOT)

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn
from wespeaker.frontend import frontend_class_dict
from wespeaker.models.ssl_backend.MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_Gaussian
from wespeaker.models.projections import get_projection
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import validate_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--data_list',  required=True)
    p.add_argument('--embed_ark',  required=True)
    p.add_argument('--utt2spk',    required=True)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers',type=int, default=2)
    p.add_argument('--crop_sec',   type=float, default=2.0,
                   help='fixed crop length in seconds (0 = whole utterance)')
    p.add_argument('--seed',       type=int, default=716)
    return p.parse_args()


def load_utt2spk(path):
    utt2spk = {}
    with open(path) as f:
        for line in f:
            utt, spk = line.strip().split(None, 1)
            utt2spk[utt] = spk
    spks = sorted(set(utt2spk.values()))
    spk2id = {s: i for i, s in enumerate(spks)}
    return utt2spk, spk2id


def main():
    args = parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # ── Build model ──────────────────────────────────────────────────────────
    model_args = configs['model_args']
    proj_args  = configs['projection_args'].copy()

    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if 'projection.weight' in ckpt:
        proj_args['num_class'] = ckpt['projection.weight'].shape[0]

    backend = SSL_BACKEND_MHFA_LN_Gaussian(**model_args)
    projection = get_projection(proj_args)
    backend.add_module('projection', projection)

    # Add frontend as submodule BEFORE load_checkpoint so its weights load too
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != 'fbank':
        fe_args_key = frontend_type + '_args'
        frontend_obj = frontend_class_dict[frontend_type](
            **configs['dataset_args'][fe_args_key],
            sample_rate=configs['dataset_args']['resample_rate'])
        backend.add_module('frontend', frontend_obj)

    load_checkpoint(backend, args.model_path)
    device = torch.device('cuda')
    backend = backend.to(device).eval()
    proj_weight = backend.projection.weight  # [C, D]
    nominal_scale = proj_args['scale']
    num_class = proj_args['num_class']

    # ── Dataset ──────────────────────────────────────────────────────────────
    test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    test_conf['spec_aug']      = False
    test_conf['shuffle']       = False
    test_conf['aug_prob']      = 0.0
    test_conf['filter']        = False

    # Fixed-length crop: derive num_frms from crop_sec so every sample is the
    # same length -> batches stack cleanly and the frontend sees uniform input.
    if args.crop_sec and args.crop_sec > 0:
        frame_shift_ms = test_conf.get(
            test_conf.get('frontend', 'fbank') + '_args', {}).get(
                'frame_shift', 10)
        test_conf['num_frms'] = int(round(args.crop_sec * 1000 / frame_shift_ms))
        whole_utt = False
        print(f"Fixed crop: {args.crop_sec}s -> num_frms={test_conf['num_frms']} "
              f"(frame_shift={frame_shift_ms}ms)")
    else:
        whole_utt = True

    ds = Dataset(configs['data_type'], args.data_list, test_conf,
                 spk2id_dict={}, whole_utt=whole_utt, repeat_dataset=False)
    loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size,
                        num_workers=args.num_workers, prefetch_factor=4)

    # ── utt2spk / spk2id ─────────────────────────────────────────────────────
    utt2spk, spk2id = load_utt2spk(args.utt2spk)

    # ── Output arks ──────────────────────────────────────────────────────────
    embed_ark    = os.path.abspath(args.embed_ark)
    embed_scp    = embed_ark[:-3] + 'scp'
    variance_ark = embed_ark[:-4] + '_variance.ark'
    variance_scp = variance_ark[:-3] + 'scp'
    scale_ark    = embed_ark[:-4] + '_scale_stats.npz'

    validate_path(embed_ark)
    validate_path(variance_ark)

    # ── Accumulators for scale stats ─────────────────────────────────────────
    all_cosine_gap   = []
    all_Lambda       = []
    all_covariance   = []
    all_intra        = []
    all_unc_scale    = []
    all_eff_scale    = []
    missing_label    = 0

    eps = 1e-6

    with torch.no_grad(), \
         kaldiio.WriteHelper(f'ark,scp:{embed_ark},{embed_scp}') as ew, \
         kaldiio.WriteHelper(f'ark,scp:{variance_ark},{variance_scp}') as vw:

        frontend_type = test_conf.get('frontend', 'fbank')

        for batch in tqdm(loader, desc=os.path.basename(args.embed_ark)):
            utts = batch['key']
            if frontend_type != 'fbank':
                wavs     = batch['wav'].squeeze(1).float().to(device)
                wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                               wavs.shape[0]).to(device)
                features, _ = backend.frontend(wavs, wavs_len)
            else:
                features = batch['feat'].float().to(device)

            if test_conf.get('cmvn', True):
                features = apply_cmvn(features, **test_conf.get('cmvn_args', {}))

            outputs    = backend(features)
            embedding  = outputs[-1]          # [B, D]
            covariance = outputs[1]           # [B, D]

            # ── Save embedding + variance ────────────────────────────────────
            for i, utt in enumerate(utts):
                ew(utt, embedding[i].cpu().numpy())
                vw(utt, covariance[i].clamp_min(eps).cpu().numpy())

            # ── Scale stats (only for utts with known speaker label) ─────────
            labels = []
            valid_idx = []
            for i, utt in enumerate(utts):
                spk = utt2spk.get(utt)
                if spk is not None and spk in spk2id:
                    labels.append(spk2id[spk])
                    valid_idx.append(i)
                else:
                    missing_label += 1

            if not valid_idx:
                continue

            vi  = torch.tensor(valid_idx, device=device)
            emb = embedding[vi]
            cov = covariance[vi]
            lbl = torch.tensor(labels, dtype=torch.long, device=device)

            cosine = F.linear(F.normalize(emb), F.normalize(proj_weight))
            cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)

            target_cos    = cosine.gather(1, lbl.view(-1, 1))
            mask          = F.one_hot(lbl, num_classes=num_class).bool()
            max_non_target = cosine.masked_fill(mask, float('-inf')).max(
                                 dim=1, keepdim=True).values

            cosine_gap = target_cos - max_non_target
            intra      = torch.exp(cosine.max(dim=1, keepdim=True).values)
            Lambda     = -cosine_gap + 0.5
            cov_denom  = (cov + Lambda + eps).clamp_min(eps)
            numer      = emb.square().sum(dim=1, keepdim=True)
            denom      = (emb.square() * cov_denom).sum(
                             dim=1, keepdim=True).clamp_min(eps)
            unc_scale  = torch.sqrt(numer / denom)
            full_mult  = unc_scale * torch.exp(cosine_gap * intra)
            eff_scale  = (1.0 + (full_mult - 1.0)) * nominal_scale

            all_cosine_gap.append(cosine_gap.squeeze(1).cpu().numpy())
            all_Lambda.append(Lambda.squeeze(1).cpu().numpy())
            all_covariance.append(cov.mean(dim=1).cpu().numpy())
            all_intra.append(intra.squeeze(1).cpu().numpy())
            all_unc_scale.append(unc_scale.squeeze(1).cpu().numpy())
            all_eff_scale.append(eff_scale.squeeze(1).cpu().numpy())

    # ── Save and print scale stats ───────────────────────────────────────────
    if all_eff_scale:
        stats = {
            'cosine_gap':      np.concatenate(all_cosine_gap),
            'Lambda':          np.concatenate(all_Lambda),
            'covariance_mean': np.concatenate(all_covariance),
            'intra':           np.concatenate(all_intra),
            'unc_scale':       np.concatenate(all_unc_scale),
            'eff_scale':       np.concatenate(all_eff_scale),
        }
        np.savez(scale_ark, **stats)

        def row(name, a):
            return (f"  {name:<22}"
                    f"{a.min():>9.4f}{np.percentile(a,25):>9.4f}"
                    f"{np.median(a):>9.4f}{np.percentile(a,75):>9.4f}"
                    f"{np.percentile(a,95):>9.4f}{a.max():>9.4f}"
                    f"{a.mean():>9.4f}{a.std():>9.4f}")

        hdr = f"{'':>24}{'min':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p95':>9}{'max':>9}{'mean':>9}{'std':>9}"
        print(f"\n{'='*90}")
        print(f"  {os.path.basename(args.embed_ark)}  n={len(stats['eff_scale'])}  "
              f"missing_label={missing_label}")
        print(hdr)
        for k, v in stats.items():
            print(row(k, v))

    print(f"\nEmbedding ark : {embed_ark}")
    print(f"Variance  ark : {variance_ark}")
    print(f"Scale stats   : {scale_ark}")


if __name__ == '__main__':
    main()
