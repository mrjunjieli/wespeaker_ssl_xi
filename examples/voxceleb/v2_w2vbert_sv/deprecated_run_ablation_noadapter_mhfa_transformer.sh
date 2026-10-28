#!/bin/bash
echo 'DEPRECATED: scalar-per-head variance baseline; use run_gaussian_diag_transformer_pos.sh or run_gaussian_diag_transformer_nopos.sh.' >&2
set -e -o pipefail

# No layer adapter; shared temporal Transformer before Gaussian MHFA pooling.
# Recipe and scoring match run_ablation_noadapter_mhfa.sh (30/3/3 epochs).
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./path.sh || exit 1

stage=1
stop_stage=15
data=data
data_type="shard"
exp="exp/v2_w2vbert_sv/mhfa_ln_gaussian_noadapter_transformer"
stage1_config=conf/mhfa_gaussian_logprec_noadapter_transformer_stage1.yaml
stage1_exp="${exp}_stage1"
stage1_eval_exp=${stage1_exp}/eval
stage1_num_avg=4
stage2_config=conf/mhfa_gaussian_logprec_noadapter_transformer_stage2_joint.yaml
stage2_exp="${exp}_stage2_joint"
stage2_eval_exp=${stage2_exp}/eval
stage2_num_avg=1
stage3_config=conf/mhfa_gaussian_logprec_noadapter_transformer_stage3_lmft.yaml
stage3_exp="${exp}_stage3_lmft_gap_margin_0.4"
stage3_eval_exp=${stage3_exp}/eval
stage3_num_avg=1
gpus="[2,3]"
checkpoint=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"
top_n=300

. tools/parse_options.sh || exit 1

# Training uses logical IDs; extraction scripts accept physical IDs.
case "$gpus" in
  "[2,3]") export CUDA_VISIBLE_DEVICES=2,3; train_gpus="[0,1]"; num_gpus=2 ;;
  "[3,2]") export CUDA_VISIBLE_DEVICES=3,2; train_gpus="[0,1]"; num_gpus=2 ;;
  "[2]") export CUDA_VISIBLE_DEVICES=2; train_gpus="[0]"; num_gpus=1 ;;
  "[3]") export CUDA_VISIBLE_DEVICES=3; train_gpus="[0]"; num_gpus=1 ;;
  *) echo "ERROR: --gpus must select physical GPUs 2 and/or 3." >&2; exit 1 ;;
esac

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  echo "Stage 1: frozen-frontend Transformer + MHFA training ..."
  python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$num_gpus" wespeaker/bin/train.py \
    --config "$stage1_config" --exp_dir "$stage1_exp" --gpus "$train_gpus" \
    --num_avg "$stage1_num_avg" --data_type "$data_type" \
    --train_data "${data}/vox2_dev/${data_type}.list" \
    --train_label "${data}/vox2_dev/utt2spk" \
    --reverb_data "${data}/rirs/lmdb" --noise_data "${data}/musan/lmdb" \
    ${checkpoint:+--checkpoint "$checkpoint"}
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  echo "Stage 2: average stage-1 checkpoints ..."
  python wespeaker/bin/average_model.py \
    --dst_model "${stage1_exp}/models/avg_model.pt" \
    --src_path "${stage1_exp}/models" --num "$stage1_num_avg"
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  echo "Stage 3: extract stage-1 embeddings and uncertainty ..."
  mkdir -p "$stage1_eval_exp"
  cp "${stage1_exp}/config.yaml" "${stage1_eval_exp}/config.yaml"
  local/extract_vox_uncertainty.sh \
    --exp_dir "$stage1_eval_exp" --model_path "${stage1_exp}/models/avg_model.pt" \
    --nj 8 --nj_vox2 16 --gpus "$gpus" --data_type "$data_type" --data "$data"
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
  echo "Stage 4: score stage-1 ordinary and uncertainty-aware trials ..."
  local/score.sh --stage 1 --stop-stage 2 --data "$data" \
    --exp_dir "$stage1_eval_exp" --trials "$trials"
  local/score_uncertainty.sh --stage 1 --stop-stage 2 --data "$data" \
    --exp_dir "$stage1_eval_exp" --trials "$trials"
fi

if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
  echo "Stage 5: initialize stage-2 from this experiment's stage-1 average ..."
  [ -f "${stage1_exp}/models/avg_model.pt" ] || \
    { echo "ERROR: stage-1 avg_model.pt missing; run stage 2 first." >&2; exit 1; }
  mkdir -p "${stage2_exp}/models"
  cp "${stage1_exp}/models/avg_model.pt" "${stage2_exp}/models/model_0.pt"
fi

if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
  echo "Stage 6: joint fine-tuning ..."
  [ -f "${stage2_exp}/models/model_0.pt" ] || \
    { echo "ERROR: stage-2 model_0.pt missing; run stage 5 first." >&2; exit 1; }
  python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$num_gpus" wespeaker/bin/train.py \
    --config "$stage2_config" --exp_dir "$stage2_exp" --gpus "$train_gpus" \
    --num_avg "$stage2_num_avg" --data_type "$data_type" \
    --train_data "${data}/vox2_dev/${data_type}.list" \
    --train_label "${data}/vox2_dev/utt2spk" \
    --reverb_data "${data}/rirs/lmdb" --noise_data "${data}/musan/lmdb" \
    --checkpoint "${stage2_exp}/models/model_0.pt"
fi

if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
  echo "Stage 7: average stage-2 checkpoints ..."
  python wespeaker/bin/average_model.py \
    --dst_model "${stage2_exp}/models/avg_model.pt" \
    --src_path "${stage2_exp}/models" --num "$stage2_num_avg"
fi

if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
  echo "Stage 8: extract stage-2 embeddings and uncertainty ..."
  mkdir -p "$stage2_eval_exp"
  cp "${stage2_exp}/config.yaml" "${stage2_eval_exp}/config.yaml"
  local/extract_vox_uncertainty.sh \
    --exp_dir "$stage2_eval_exp" --model_path "${stage2_exp}/models/avg_model.pt" \
    --nj 8 --gpus "$gpus" --data_type "$data_type" --data "$data"
fi

if [ "$stage" -le 9 ] && [ "$stop_stage" -ge 9 ]; then
  echo "Stage 9: score stage-2 ordinary and uncertainty-aware trials ..."
  local/score.sh --stage 1 --stop-stage 2 --data "$data" \
    --exp_dir "$stage2_eval_exp" --trials "$trials"
  local/score_uncertainty.sh --stage 1 --stop-stage 2 --data "$data" \
    --exp_dir "$stage2_eval_exp" --trials "$trials"
fi

if [ "$stage" -le 10 ] && [ "$stop_stage" -ge 10 ]; then
  echo "Stage 10: initialize stage-3 from this experiment's stage-2 average ..."
  [ -f "${stage2_exp}/models/avg_model.pt" ] || \
    { echo "ERROR: stage-2 avg_model.pt missing; run stage 7 first." >&2; exit 1; }
  mkdir -p "${stage3_exp}/models"
  cp "${stage2_exp}/models/avg_model.pt" "${stage3_exp}/models/model_0.pt"
fi

if [ "$stage" -le 11 ] && [ "$stop_stage" -ge 11 ]; then
  echo "Stage 11: gap-margin fine-tuning, margin 0.2 -> 0.4 ..."
  [ -f "${stage3_exp}/models/model_0.pt" ] || \
    { echo "ERROR: stage-3 model_0.pt missing; run stage 10 first." >&2; exit 1; }
  python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$num_gpus" wespeaker/bin/train.py \
    --config "$stage3_config" --exp_dir "$stage3_exp" --gpus "$train_gpus" \
    --num_avg "$stage3_num_avg" --data_type "$data_type" \
    --train_data "${data}/vox2_dev/${data_type}.list" \
    --train_label "${data}/vox2_dev/utt2spk" \
    --reverb_data "${data}/rirs/lmdb" --noise_data "${data}/musan/lmdb" \
    --checkpoint "${stage3_exp}/models/model_0.pt"
fi

if [ "$stage" -le 12 ] && [ "$stop_stage" -ge 12 ]; then
  echo "Stage 12: average stage-3 checkpoints ..."
  python wespeaker/bin/average_model.py \
    --dst_model "${stage3_exp}/models/avg_model.pt" \
    --src_path "${stage3_exp}/models" --num "$stage3_num_avg"
fi

if [ "$stage" -le 13 ] && [ "$stop_stage" -ge 13 ]; then
  echo "Stage 13: extract stage-3 embeddings and uncertainty ..."
  mkdir -p "$stage3_eval_exp"
  cp "${stage3_exp}/config.yaml" "${stage3_eval_exp}/config.yaml"
  local/extract_vox_uncertainty.sh \
    --exp_dir "$stage3_eval_exp" --model_path "${stage3_exp}/models/avg_model.pt" \
    --nj 16 --gpus "$gpus" --data_type "$data_type" --data "$data"
fi

if [ "$stage" -le 14 ] && [ "$stop_stage" -ge 14 ]; then
  echo "Stage 14: score stage-3 ordinary and uncertainty-aware trials ..."
  local/score.sh --stage 1 --stop-stage 2 --data "$data" \
    --exp_dir "$stage3_eval_exp" --trials "$trials"
  local/score_uncertainty.sh --stage 1 --stop-stage 2 --data "$data" \
    --exp_dir "$stage3_eval_exp" --trials "$trials"
fi

if [ "$stage" -le 15 ] && [ "$stop_stage" -ge 15 ]; then
  echo "Stage 15: stage-3 uncertainty-aware AS-norm ..."
  local/score_norm_uncertainty.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method "$score_norm_method" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "$data" --exp_dir "$stage3_eval_exp" --trials "$trials"
fi
