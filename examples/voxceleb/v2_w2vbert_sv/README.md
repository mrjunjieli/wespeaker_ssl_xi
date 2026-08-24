# W2V-BERT 2.0 + MHFA-LN-U-Cube speaker verification

This recipe implements uncertainty-aware speaker verification with a W2V-BERT
2.0 frontend, an MHFA-LN-U-Cube backend, and a two-phase LoRA-to-full-fine-
tuning workflow. Training and extraction are orchestrated by `run.sh`.

## Modification attribution and provenance

This extension was developed by **Junjie Li**
([junjie98.li@connect.polyu.hk](mailto:junjie98.li@connect.polyu.hk)) on top of
the existing WeSpeaker and W2V-BERT speaker-verification implementation.
Original author attributions and Apache License 2.0 notices remain preserved in
the corresponding source files.

The main additions are:

- independent MHFA Key/Value fusion over all W2V-BERT hidden states;
- separate Key/Value compression and LayerNorm;
- U-Cube-inspired frame-level precision estimation and Gaussian posterior
  pooling;
- propagation of posterior variance to the speaker-embedding space;
- inter/intra uncertainty-aware ArcMargin with a curriculum weight;
- Q/K LoRA training with a differential frontend learning rate; and
- Stage 1 LoRA training, LoRA merge/unload, then Stage 2 full frontend
  fine-tuning.

The recipe has **two training phases**, exposed as **six orchestration stages**
in `run.sh`. The stage numbers below refer to `run.sh`, not six independent
training phases.

## Model architecture

```text
waveform [B, samples]
  |
  v
W2V-BERT 2.0 frontend
  24 Transformer encoder layers, hidden size 1024
  |
  v
25 hidden-state representations [B, 1024, T, 25]
  |
  +-- learned Key layer weights  --> Linear(1024,128) --> LayerNorm
  |
  +-- learned Value layer weights --> Linear(1024,128) --> LayerNorm
  |                                      [B,T,128]
  v
U-Cube precision estimator (from Key)
  Conv1d(128,256) --> ReLU --> BatchNorm
  --> 8-head MultiView Transformer --> Conv1d(256,128) --> Softplus
  |
  v
frame log-precision [B,T,128]
  + Value features [B,T,128]
  + learned Gaussian prior
  |
  v
one Gaussian posterior
  posterior mean     [B,128]
  posterior variance [B,128]
  |
  v
Linear(128,256)
  speaker embedding             [B,256]
  propagated embedding variance [B,256]
```

The original MHFA 64 pooling heads are not used by this backend. The retained
8 heads belong to the MultiView temporal precision encoder; they estimate
context for one frame/feature precision map rather than producing 8 or 64
separate speaker embeddings.

The backend forward output is:

```text
(frame_values [B,T,128], embedding_variance [B,256], embedding [B,256])
```

During extraction, the final tuple element is written as the speaker embedding.

## Current Stage 1 configuration

Source: `conf/w2vbert_sv_mhfa_stage1_frozen.yaml`.

| Setting | Current value |
|---|---|
| Frontend | W2V-BERT 2.0 base frozen; Q/K LoRA trainable |
| LoRA | `r=8`, `alpha=16`, targets `linear_q`/`linear_k`, dropout `0.1`, bias `none` |
| Backend | `SSL_BACKEND_MHFA_LN_UCube` |
| Projection | `arc_margin_uncertainty`, scale 32 |
| Epochs / checkpoint interval | 40 / 5 |
| Batch size / AMP | 64 / enabled |
| Segment length | 150 frames |
| Speed perturbation | enabled |
| MUSAN/RIR augmentation probability | 0.6 |
| Margin curriculum | 0.0 to 0.2, epochs 10-25, linear |
| Uncertainty curriculum | weight 0.0 to 1.0, epochs 2-5, linear |
| Optimizer | AdamW, weight decay `1e-7` |
| LR schedule | exponential decrease, `1e-4` to `1e-5`, 2 warm-up epochs |
| LoRA/frontend LR | `0.5 x` backend LR |

With speed perturbation, 5,994 VoxCeleb2 speakers become 17,982 training
classes. The current trainable parameter counts are:

| Trainable component | Parameters |
|---|---:|
| W2V-BERT Q/K LoRA, rank 8 | 786,432 (0.786432 M) |
| MHFA-LN-U-Cube backend | 1,152,434 (1.152434 M) |
| 17,982-class, 256-dimensional classifier | 4,603,392 (4.603392 M) |
| **Stage 1 total** | **6,542,258 (6.542258 M)** |

The LoRA count is:

```text
24 layers x 2 targets (Q,K) x 2 matrices (A,B) x 1024 x rank 8
= 786,432 parameters
```

`alpha/r = 16/8 = 2`. The frozen W2V-BERT base is still executed in forward
and backward graphs so gradients can reach LoRA adapters in every layer;
parameter-efficient LoRA therefore does not remove the activation cost of the
24-layer frontend.

## Current Stage 2 configuration

Source: `conf/w2vbert_sv_mhfa_stage2_ft.yaml`.

Before Stage 2, `wespeaker/bin/merge_w2vbert_lora.py` folds each LoRA update
into its W2V-BERT Q/K weight and unloads the PEFT wrapper. Stage 2 therefore has
no separate LoRA A/B parameters: it fine-tunes the merged W2V-BERT model.

| Setting | Current value |
|---|---|
| Frontend | merged W2V-BERT; fully trainable; `use_lora: false` |
| Backend | `SSL_BACKEND_MHFA_LN_UCube` |
| Projection | ordinary `arc_margin`, scale 32 |
| Epochs / checkpoint interval | 8 / 1 |
| Batch size / AMP | 16 / enabled |
| Segment length | 200 frames |
| Speed perturbation | disabled (5,994 classes) |
| MUSAN/RIR augmentation probability | 0.6 |
| Margin curriculum | 0.2 to 0.4, epochs 2-6, linear |
| Optimizer | AdamW, weight decay `1e-5` |
| LR schedule | exponential decrease, `5e-5` to `1e-6`, 1 warm-up epoch |
| Frontend LR | `0.5 x` backend LR |

Stage 2 intentionally uses ordinary ArcMargin rather than the uncertainty-aware
projection. The backend still estimates and returns embedding variance, but the
Stage 2 classifier consumes only the 256-dimensional embedding.

## `run.sh` stages

The current script defaults to `stage=1`, `stop_stage=1`, GPUs `[2,3]`, four
Stage 1 checkpoints for averaging, and one Stage 2 checkpoint for averaging.
Command-line options override these values.

| `run.sh` stage | Action |
|---:|---|
| 1 | Train frozen W2V-BERT base + Q/K LoRA + MHFA-LN-U-Cube + classifier |
| 2 | Average the last Stage 1 checkpoints, extract Vox2-dev/Vox1 embeddings, and score Vox1-O/E/H |
| 3 | Merge Stage 1 LoRA into W2V-BERT and run Stage 2 full-model fine-tuning |
| 4 | Average the final Stage 2 checkpoints |
| 5 | Extract Stage 2 Vox2-dev embeddings for mean normalization and Vox1 embeddings for evaluation |
| 6 | Mean-normalized cosine scoring and EER/minDCF on cleaned Vox1-O/E/H trials |

Default experiment locations are:

```text
Stage 1: exp/v2_w2vbert_sv/stage1_frozen_vox2
Stage 1 evaluation: exp/v2_w2vbert_sv/stage1_frozen_vox2/eval
Stage 2: exp/v2_w2vbert_sv/stage2_ft_vox2
```

Scoring uses Vox2-dev mean normalization, cosine similarity, and reports EER
and minDCF with `p_target=0.01`, `c_miss=1`, and `c_fa=1`.

## Running the current recipe

```bash
cd examples/voxceleb/v2_w2vbert_sv

# Stage 1 training only (the run.sh default)
bash run.sh

# Stage 1 training, averaging, extraction, and evaluation
bash run.sh --stage 1 --stop_stage 2

# Merge LoRA, run Stage 2 full fine-tuning, then average its checkpoints
bash run.sh --stage 3 --stop_stage 4

# Extract and score the final Stage 2 model
bash run.sh --stage 5 --stop_stage 6
```

To resume Stage 1 from a checkpoint:

```bash
bash run.sh --stage 1 --stop_stage 1 \
  --checkpoint exp/v2_w2vbert_sv/stage1_frozen_vox2/models/model_<epoch>.pt
```

Training commands in this checkout default to physical GPUs 2 and 3. If
`CUDA_VISIBLE_DEVICES=2,3` is set externally, pass logical GPU IDs `[0,1]`
instead of `[2,3]`; otherwise CUDA reports an invalid device ordinal.

## Data and assets

Expected layout:

```text
assets/
└── w2v-bert-2.0/
    ├── config_baseline.json
    ├── model.safetensors
    └── preprocessor_config.json

data/
├── vox2_dev/
│   ├── shard.list
│   └── utt2spk
├── vox1/
│   ├── shard.list
│   ├── raw.list
│   └── trials/
│       ├── vox1_O_cleaned.kaldi
│       ├── vox1_E_cleaned.kaldi
│       └── vox1_H_cleaned.kaldi
├── musan/lmdb
└── rirs/lmdb
```

Primary VoxCeleb speech is read from shard lists. MUSAN and RIR LMDBs are
independent augmentation sources; they do not change the primary speech data
type.

`local/prepare_assets.sh` downloads public W2V-BERT/released-checkpoint assets
and trial lists. Scoring already released embeddings does not require VoxCeleb
audio, but it still requires the trial-list files.

### Known asset-preparation caveat

The current `local/prepare_assets.sh` references
`conf/w2vbert_sv_mhfa.yaml`, but that inference config is not present in the
current recipe directory. A bare-clone preparation run must provide/correct
that inference config before the released-checkpoint extraction path can be
considered self-contained. This does not affect Stage 1 or Stage 2 training,
which use the two configuration files documented above.

## Historical released-checkpoint reference

The following numbers belong to earlier released/sibling MHFA-W2V-BERT runs.
They are retained only as historical references and were **not** produced by
the current rank-8, 40-epoch Stage 1 plus 8-epoch Stage 2 defaults.

| Historical checkpoint | Vox1-O EER / minDCF | Vox1-E EER / minDCF | Vox1-H EER / minDCF |
|---|---|---|---|
| `mhfa_W2VB_ft_v2` ([HF Hub](https://huggingface.co/JYP2024/wespeaker-mhfa-W2VB-ft-v2)) | 0.218 / 0.024 | 0.378 / 0.036 | 0.770 / 0.074 |
| `mhfa_W2VB_ft_v5` (local sibling run) | 0.207 / 0.019 | 0.377 / 0.036 | 0.756 / 0.072 |

Do not use the historical three-epoch fine-tuning metadata or historical
frontend-LR behavior as a description of the active YAML files. In this repo,
`train.py` passes the model to the scheduler, so `frontend_lr_ratio` is applied.

## Main implementation files

| File | Current role |
|---|---|
| `wespeaker/frontend/get_hf_w2vbert_pruning.py` | Loads W2V-BERT, injects Q/K LoRA, freezes/unfreezes the base, and returns all hidden states |
| `wespeaker/models/ssl_backend/MHFA.py` | Implements `SSL_BACKEND_MHFA_LN_UCube`, Gaussian posterior pooling, and variance propagation |
| `wespeaker/models/ssl_backend/multi_view_attention.py` | Implements the 8-head temporal MultiView precision encoder |
| `wespeaker/models/projections.py` | Implements inter/intra uncertainty-aware ArcMargin |
| `wespeaker/utils/schedulers.py` | Implements uncertainty curriculum and differential frontend LR grouping |
| `wespeaker/utils/executor.py` | Routes backend variance to the uncertainty projection during Stage 1 |
| `wespeaker/bin/train.py` | Builds the frontend/backend/projection and active schedulers |
| `wespeaker/bin/merge_w2vbert_lora.py` | Merges Stage 1 LoRA weights and unloads PEFT for Stage 2 |
| `examples/voxceleb/v2_w2vbert_sv/run.sh` | Orchestrates training, merge, averaging, extraction, and scoring |

The frontend key is `w2vbert_sv`; configs for a different upstream frontend
key are not interchangeable without adapting their argument block.
