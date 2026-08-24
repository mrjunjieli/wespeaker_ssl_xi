# W2V-BERT 2.0 + MHFA speaker verification (ported from wespeaker_hubert)

Self-contained recipe: model weights live in `assets/`, data lists + LMDB
links in `data/` (both git-ignored — see layout below), all stage logic in
`run.sh`. On a machine where `assets/` and `data/` are populated (e.g. the
LUMI checkout), everything below works as-is.

## Modification attribution and provenance

This extension was developed by **Junjie Li**
([junjie98.li@connect.polyu.hk](mailto:junjie98.li@connect.polyu.hk)) on top
of the existing WeSpeaker and W2V-BERT speaker-verification implementation.
All original author attributions and the Apache License 2.0 notices are
preserved in the corresponding source files.

The main additions in this version are:

- an MHFA-LN + U-Cube backend with a single Gaussian posterior and propagated
  diagonal embedding variance;
- inter/intra uncertainty-aware ArcMargin with an independent curriculum;
- rank-12 LoRA adapters on W2V-BERT `linear_q` and `linear_k`, including
  differential frontend/backend learning rates; and
- a two-stage workflow that trains the frozen base plus LoRA and backend in
  Stage 1, merges LoRA into W2V-BERT, then performs full fine-tuning in Stage 2.

These changes extend rather than replace the prior implementation. See the
copyright headers in the modified source files for both the new modification
author and the original authors.

## Quick start

```bash
cd examples/voxceleb/v2_w2vbert_sv

# (0) on a bare clone: fetch base model + released checkpoint from the HF Hub
bash local/prepare_assets.sh

# (a) score the released reference embeddings (no GPU, minutes)
bash run.sh --stage 6 --stop_stage 6 --stage2_exp assets/mhfa_W2VB_ft_v2

# (b) evaluate the released fine-tuned checkpoint from scratch (GPU)
bash run.sh --stage 5 --stop_stage 6 --stage2_exp assets/mhfa_W2VB_ft_v2

# (c) full two-stage training (GPU, multi-day)
bash run.sh --stage 1 --stop_stage 2   # stage-1: frozen-frontend training + average + evaluation
bash run.sh --stage 3 --stop_stage 4   # stage-2: full fine-tuning + average
bash run.sh --stage 5 --stop_stage 6   # extract embeddings + score vox1-O/E/H
```

The stage-1 average is evaluated immediately after stage 2. Its embeddings and
scores are stored under `stage1_exp/eval/`, while the final stage-2 evaluation
uses `stage2_exp/`; compare the corresponding `scores/vox1_cos_result` files.
Training and extraction use the configured shard lists; MUSAN and RIR LMDBs
are used only for training-time augmentation. On a cluster, wrap `run.sh` in
your scheduler submission script.

## Current uncertainty-aware training architecture

The training configs now use `SSL_BACKEND_MHFA_LN_UCube`. It retains the
independent MHFA key/value layer fusion and LayerNorm projections, replaces the
original 64 pooling heads with one U-Cube Gaussian posterior, and returns a
256-dimensional embedding with its diagonal variance. The classifier is
`arc_margin_uncertainty`.

Stage 1 freezes the W2V-BERT base and trains rank-12 LoRA adapters on all 24
attention `linear_q` and `linear_k` projections together with the backend. This
adds 1,179,648 trainable frontend parameters, close to the 1,152,434 backend
parameters (classifier excluded). Before Stage 2, `merge_w2vbert_lora.py` folds
the adapters into the attention weights; Stage 2 then loads the merged
checkpoint without PEFT wrappers and fully fine-tunes W2V-BERT.

## Results (cosine + vox2_dev mean-norm)

| Checkpoint | vox1-O EER / minDCF | vox1-E | vox1-H |
|---|---|---|---|
| **mhfa_W2VB_ft_v2** (released, [HF Hub](https://huggingface.co/JYP2024/wespeaker-mhfa-W2VB-ft-v2)) | 0.218 / 0.024 | 0.378 / 0.036 | 0.770 / 0.074 |
| mhfa_W2VB_ft_v5 (local-only sibling run) | 0.207 / 0.019 | 0.377 / 0.036 | 0.756 / 0.072 |

Both are 3-epoch full fine-tunes of the same stage-1 recipe (different
seeds); ft_v2 is the publicly downloadable release, ft_v5 exists only on the
LUMI checkout. Numbers reproduced 2026-08-10 with quick-start command (a).

## Layout of the git-ignored resources

```
assets/                           # rebuild anywhere with local/prepare_assets.sh
├── w2v-bert-2.0/                 # base model: config_baseline.json + model.safetensors
└── mhfa_W2VB_ft_v2/              # released two-stage result (vox2_dev, 3 ep ft)
    ├── config.yaml               # inference config (this repo's format)
    ├── models/avg_model.pt
    └── embeddings/{vox1,vox2_dev}  # reference embeddings (scp+ark)
data/
├── vox2_dev/  shard.list utt2spk wav.scp spk2utt
├── vox1/      shard.list utt2spk wav.scp spk2utt trials/
├── musan/     lmdb -> (musan aug lmdb)
└── rirs/      lmdb -> (rirs aug lmdb)
```

The VoxCeleb training and extraction datasets are read from `shard.list`.
MUSAN and RIR LMDBs are separate augmentation sources and do not control how
the primary speech data are loaded.

## Reproducing from a bare clone

1. **Environment**: `pip install -r requirements.txt` plus `torchnet`
   (needed by the training executor, missing from upstream requirements).
2. **Models** (fully public): `bash local/prepare_assets.sh` downloads the
   official `facebook/w2v-bert-2.0` base (our copy is bit-identical, sha256
   `eb890c96...`), installs `conf/config_baseline.json` (the official config
   plus additive pruning fields, shipped in this repo), and fetches the
   released `mhfa_W2VB_ft_v2` checkpoint + reference embeddings from
   [JYP2024/wespeaker-mhfa-W2VB-ft-v2](https://huggingface.co/JYP2024/wespeaker-mhfa-W2VB-ft-v2).
   Quick-start (a) then reproduces the results table with no data at all;
   (b) additionally needs the vox1/vox2_dev audio.
3. **Data** (bring your own, licenses forbid redistribution): VoxCeleb1/2 +
   MUSAN + RIRs. Build the standard WeSpeaker shard layout with the upstream
   tooling (`examples/voxceleb/v2/local/prepare_data.sh` and
   `tools/make_shard_list.py`), place or link it as `data/` here, and build
   LMDBs only for MUSAN/RIR augmentation. `data/vox1/trials` contains the
   official cleaned VoxCeleb trial lists.
4. **Retraining** (optional): `run.sh` stages 1-4, two-stage recipe as
   configured; note the `frontend_lr_ratio` caveat below.

## What was integrated into wespeaker (minimal diff to upstream)

| Added / changed | What |
|---|---|
| `wespeaker/frontend/wav2vecbert/` | Custom HF Wav2Vec2-BERT modeling code (pruning-capable, HardConcrete) |
| `wespeaker/frontend/get_hf_w2vbert_pruning.py` | `HuggingfaceFrontend_w2vbert` frontend wrapper |
| `wespeaker/frontend/__init__.py` | Registers frontend key **`w2vbert_sv`** (lazy import); guards optional `s3prl` / `whisper` imports |
| `wespeaker/models/ssl_backend/` | `SSL_BACKEND_MHFA` / `SSL_BACKEND_MHFA_LayerNorm` backends |
| `wespeaker/models/speaker_model.py` | Dispatches `SSL_BACKEND*` model names |
| `wespeaker/utils/schedulers.py` | BaseClass schedulers support `frontend_modules` / `frontend_lr_ratio` (differential frontend lr) |
| `wespeaker/bin/train.py` | Passes the model to BaseClass schedulers so the differential lr actually applies |
| `wespeaker/dataset/` + bin + `tools/extract_embedding.sh` | Supports shard-based training and extraction; MUSAN/RIR augmentation remains LMDB-based |

Note: the frontend key is `w2vbert_sv`, NOT `w2vbert` — the upstream
`w2vbert` key is the LoRA-adapter variant from wespeaker PR #439. Configs
written for the original recipe (`frontend: w2vbert`, `w2vbert_args:`) must
be renamed accordingly (see `conf/`). Training is launched via
`python -m torch.distributed.run` (not the `torchrun` binary) so worker
processes inherit the active interpreter in venv-over-conda containers.

Historical caveat: in the original v4_pruning runs `frontend_lr_ratio: 0.1`
was silently inactive (the scheduler never received the model reference), so
the released ft_v2/ft_v5 checkpoints were effectively trained with FULL
frontend lr. In this repo the setting works; use `frontend_lr_ratio: 1.0` to
replicate the historical runs exactly.

## Equivalence to the original recipe

Verified 2026-08-10 (PyTorch 2.6.0 rocm container, MI250X, fp32): both
checkpoints load with 0 missing / 0 unexpected keys, and 16 vox1 utterances
(12 dev + 4 test) extracted through this repo match the corresponding
reference arks at float32 GPU-reproducibility level — worst-case relative
diff 3.1e-06 (ft_v2, vs its HF reference embeddings) and 2.9e-06 (ft_v5),
cosine 1.00000000, identically via the disk and LMDB input paths. Check any
new extraction against the reference:

```bash
python local/compare_embeddings.py \
  <new_xvector.scp> assets/mhfa_W2VB_ft_v2/embeddings/vox1/xvector.scp
```
