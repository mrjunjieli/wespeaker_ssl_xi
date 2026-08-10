# W2V-BERT 2.0 + MHFA speaker verification (ported from wespeaker_hubert)

This example integrates the custom W2V-BERT 2.0 SV stack
(`third_party/w2vbert_sv` of the spkreasoner project, originally trained with
the `wespeaker_hubert` recipe) into mainline WeSpeaker with minimal changes:

| Added / changed | What |
|---|---|
| `wespeaker/frontend/wav2vecbert/` | Custom HF Wav2Vec2-BERT modeling code (pruning-capable, HardConcrete) |
| `wespeaker/frontend/get_hf_w2vbert_pruning.py` | `HuggingfaceFrontend_w2vbert` frontend wrapper |
| `wespeaker/frontend/__init__.py` | Registers frontend key **`w2vbert_sv`** (lazy import); guards optional `s3prl` / `whisper` imports |
| `wespeaker/models/ssl_backend/` | `SSL_BACKEND_MHFA` / `SSL_BACKEND_MHFA_LayerNorm` backends |
| `wespeaker/models/speaker_model.py` | Dispatches `SSL_BACKEND*` model names |
| `wespeaker/utils/schedulers.py` | BaseClass schedulers support `frontend_modules` / `frontend_lr_ratio` (differential frontend lr) |
| `wespeaker/bin/train.py` | Passes the model to BaseClass schedulers so the differential lr actually applies |
| `wespeaker/dataset/` + bin + `tools/extract_embedding.sh` | Optional `train_lmdb` (raw audio from LMDB, keyed by the raw.list `wav` field); default behavior unchanged |

Note: the frontend key is `w2vbert_sv`, NOT `w2vbert` — the upstream `w2vbert`
key is the LoRA-adapter variant from wespeaker PR #439. Configs written for the
original recipe (`frontend: w2vbert`, `w2vbert_args:`) must be renamed to
`frontend: w2vbert_sv`, `w2vbert_sv_args:` (see `conf/w2vbert_sv_mhfa.yaml`).

## Assets (LUMI paths, not in git)

```
/scratch/project_465002316/junyi/spkreasoner/third_party/w2vbert_sv/
├── convert_w2vbert/w2v-bert-2.0/     # base model (config_baseline.json + model.safetensors)
├── mhfa_W2VB_ft_v5/models/avg_model.pt   # fine-tuned checkpoint (vox2_dev, 3 ep)
└── mhfa_W2VB_ft_v5/embeddings/       # reference vox1/vox2_dev embeddings
```

## Two-stage training (vox2_dev recipe)

`conf/w2vbert_sv_mhfa_stage1_frozen.yaml` (frozen frontend, 15 ep) and
`conf/w2vbert_sv_mhfa_stage2_ft.yaml` (full fine-tune, 3 ep) are the ported
v4_pruning two-stage configs. Handoff between stages as in the original
recipe: copy stage-1 `models/avg_model.pt` to stage-2 `models/model_0.pt` and
launch stage-2 training with `--checkpoint <stage2_exp>/models/model_0.pt`.
Data args (`--train_data raw.list --train_label utt2spk --reverb_data
--noise_data`) go on the command line. Raw audio can come from disk (default)
or from an LMDB via the optional `--train_lmdb <lmdb_dir>` (the LMDB maps the
raw.list `wav` value to the original wav bytes; also available in
`tools/extract_embedding.sh` as `--data_lmdb`). A complete, still-alive
vox2_dev data dir (raw.list + utt2spk + lmdb, 1,092,009 utts, plus
musan/rirs lmdb) lives at
`/scratch/project_465002316/junyi/sv/wespeaker_hubert/examples/voxceleb/v4/data/`.

Caveat discovered while porting: in the original v4_pruning runs the
`frontend_lr_ratio: 0.1` setting was silently inactive (train.py never passed
the model to the scheduler, and frontend detection needs it), so the released
`mhfa_W2VB_ft_v5` checkpoint was effectively trained with FULL frontend lr.
In this repo the setting works; use `frontend_lr_ratio: 1.0` to replicate the
historical run exactly.

## Extract embeddings

```bash
export PYTHONPATH=$REPO:$REPO/.deps   # .deps: silero-vad/fire/onnxruntime etc. (pip install --no-deps --target .deps)
python wespeaker/bin/extract.py \
  --config examples/voxceleb/v2_w2vbert_sv/conf/w2vbert_sv_mhfa.yaml \
  --model_path .../mhfa_W2VB_ft_v5/models/avg_model.pt \
  --data_type raw --data_list raw.list \
  --embed_ark xvector.ark --batch_size 1 --num_workers 2
```

`raw.list` is standard WeSpeaker JSON-lines: `{"key": ..., "spk": ..., "wav": ...}`.

## Equivalence check against the original recipe

`local/compare_embeddings.py` compares a freshly extracted scp/ark against the
reference embeddings shipped with the checkpoint:

```bash
python examples/voxceleb/v2_w2vbert_sv/local/compare_embeddings.py \
  new_xvector.scp .../mhfa_W2VB_ft_v5/embeddings/vox1/xvector.scp
```

Verified 2026-08-10 on LUMI (PyTorch 2.6.0 rocm-6.2.4 container, dev-g MI250X):
16 vox1 utterances (12 dev + 4 test), checkpoint loads with 0 missing /
0 unexpected keys; extracted embeddings match the reference arks with
worst-case relative diff 2.9e-06 and cosine 1.00000000 (float32 GPU
reproducibility level).
