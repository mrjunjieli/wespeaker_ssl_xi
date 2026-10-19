#!/bin/bash

# Three-stage recipe for W2V-BERT 2.0 + MHFA-LN Gaussian log-precision:
#   stage 1: frozen-frontend backend training           (fixed, unchanged)
#   stage 2: joint fine-tuning, 3s chunks + stage-1 augmentation, margin 0.2
#   stage 3: large-margin fine-tuning (LMFT), 5s chunks, with augmentation,
#            margin 0.2 -> 0.4

. ./path.sh || exit 1

stage=1
stop_stage=15
data=data
data_type="shard"


exp="exp/v2_w2vbert_sv/mhfa_ln_guassian_lightadapter"
# Stage 1: frozen frontend training (fixed, do not change)
stage1_config=conf/mhfa_gaussian_logprec_stage1.yaml
stage1_exp="$exp"_stage1
stage1_eval_exp=${stage1_exp}/eval
stage1_num_avg=4

# Stage 2: joint fine-tuning (3s chunks + stage-1 augmentation, margin 0.2)
stage2_config=conf/mhfa_gaussian_logprec_stage2_joint.yaml
stage2_exp="$exp"_stage2_joint
stage2_eval_exp=${stage2_exp}/eval
stage2_num_avg=1

# Stage 3: large-margin fine-tuning (5s chunks, with augmentation, margin 0.2->0.4)
#   target-gap adaptive per-sample margin (gap is detached)
#   direction: smaller gap (harder) -> larger margin (margin = base + span*z)
stage3_config=conf/mhfa_gaussian_logprec_stage3_lmft.yaml
stage3_exp="$exp"_stage3_lmft_gap_margin_0.4
stage3_eval_exp=${stage3_exp}/eval
stage3_num_avg=1

gpus="[2,3]"
checkpoint=
# checkpoint=$stage1_exp/models/model_5.pt
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

. tools/parse_options.sh || exit 1 

num_gpus=$(echo "$gpus" | awk -F ',' '{print NF}')

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: train stage-1 frozen-frontend backend ..."
  python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=$num_gpus wespeaker/bin/train.py \
    --config $stage1_config --exp_dir ${stage1_exp} --gpus $gpus \
    --num_avg ${stage1_num_avg} --data_type ${data_type} \
    --train_data ${data}/vox2_dev/${data_type}.list \
    --train_label ${data}/vox2_dev/utt2spk \
    --reverb_data ${data}/rirs/lmdb --noise_data ${data}/musan/lmdb \
    ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: average stage-1 checkpoints ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage1_exp}/models/avg_model.pt \
    --src_path ${stage1_exp}/models --num ${stage1_num_avg}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: extract stage-1 embeddings and uncertainty ..."
  mkdir -p ${stage1_eval_exp}
  cp ${stage1_exp}/config.yaml ${stage1_eval_exp}/config.yaml
  local/extract_vox_uncertainty.sh \
    --exp_dir ${stage1_eval_exp} --model_path ${stage1_exp}/models/avg_model.pt \
    --nj 8 --nj_vox2 16 --gpus "$gpus" --data_type ${data_type} --data ${data}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: score stage-1 ordinary and uncertainty-aware trials ..."
  local/score.sh --stage 1 --stop-stage 2 --data ${data} \
    --exp_dir ${stage1_eval_exp} --trials "$trials"
  local/score_uncertainty.sh --stage 1 --stop-stage 2 --data ${data} \
    --exp_dir ${stage1_eval_exp} --trials "$trials"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: copy stage-1 avg model as stage-2 init ..."
  [ -f "${stage1_exp}/models/avg_model.pt" ] || \
    { echo "ERROR: ${stage1_exp}/models/avg_model.pt not found — run stage 2 first." >&2; exit 1; }
  mkdir -p ${stage2_exp}/models
  cp "${stage1_exp}/models/avg_model.pt" "${stage2_exp}/models/model_0.pt"
  [ -f "${stage2_exp}/models/model_0.pt" ] || \
    { echo "ERROR: copy failed — ${stage2_exp}/models/model_0.pt not found." >&2; exit 1; }
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: joint fine-tuning from stage-1 checkpoint ..."
  [ -f "${stage2_exp}/models/model_0.pt" ] || \
    { echo "ERROR: ${stage2_exp}/models/model_0.pt not found — run stage 5 first." >&2; exit 1; }
  python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=$num_gpus wespeaker/bin/train.py \
    --config $stage2_config --exp_dir ${stage2_exp} --gpus $gpus \
    --num_avg ${stage2_num_avg} --data_type ${data_type} \
    --train_data ${data}/vox2_dev/${data_type}.list \
    --train_label ${data}/vox2_dev/utt2spk \
    --reverb_data ${data}/rirs/lmdb --noise_data ${data}/musan/lmdb \
    --checkpoint ${stage2_exp}/models/model_0.pt
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: average stage-2 checkpoints ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage2_exp}/models/avg_model.pt \
    --src_path ${stage2_exp}/models --num ${stage2_num_avg}
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Stage 8: extract stage-2 embeddings and uncertainty ..."
  mkdir -p ${stage2_eval_exp}
  cp ${stage2_exp}/config.yaml ${stage2_eval_exp}/config.yaml
  local/extract_vox_uncertainty.sh \
    --exp_dir ${stage2_eval_exp} --model_path ${stage2_exp}/models/avg_model.pt \
    --nj 8 --gpus "$gpus" --data_type ${data_type} --data ${data}
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Stage 9: score stage-2 ordinary and uncertainty-aware trials ..."
  local/score.sh --stage 1 --stop-stage 2 --data ${data} \
    --exp_dir ${stage2_eval_exp} --trials "$trials"
  local/score_uncertainty.sh --stage 1 --stop-stage 2 --data ${data} \
    --exp_dir ${stage2_eval_exp} --trials "$trials"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "Stage 10: copy stage-2 avg model as stage-3 init ..."
  [ -f "${stage2_exp}/models/avg_model.pt" ] || \
    { echo "ERROR: ${stage2_exp}/models/avg_model.pt not found — run stage 7 first." >&2; exit 1; }
  mkdir -p ${stage3_exp}/models
  cp "${stage2_exp}/models/avg_model.pt" "${stage3_exp}/models/model_0.pt"
  [ -f "${stage3_exp}/models/model_0.pt" ] || \
    { echo "ERROR: copy failed — ${stage3_exp}/models/model_0.pt not found." >&2; exit 1; }
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "Stage 11: large-margin fine-tuning (LMFT) from stage-2 checkpoint ..."
  [ -f "${stage3_exp}/models/model_0.pt" ] || \
    { echo "ERROR: ${stage3_exp}/models/model_0.pt not found — run stage 10 first." >&2; exit 1; }
  python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node=$num_gpus wespeaker/bin/train.py \
    --config $stage3_config --exp_dir ${stage3_exp} --gpus $gpus \
    --num_avg ${stage3_num_avg} --data_type ${data_type} \
    --train_data ${data}/vox2_dev/${data_type}.list \
    --train_label ${data}/vox2_dev/utt2spk \
    --reverb_data ${data}/rirs/lmdb --noise_data ${data}/musan/lmdb \
    --checkpoint ${stage3_exp}/models/model_0.pt
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "Stage 12: average stage-3 checkpoints ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage3_exp}/models/avg_model.pt \
    --src_path ${stage3_exp}/models --num ${stage3_num_avg}
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  echo "Stage 13: extract stage-3 embeddings and uncertainty ..."
  mkdir -p ${stage3_eval_exp}
  cp ${stage3_exp}/config.yaml ${stage3_eval_exp}/config.yaml
  local/extract_vox_uncertainty.sh \
    --exp_dir ${stage3_eval_exp} --model_path ${stage3_exp}/models/avg_model.pt \
    --nj 16 --gpus "$gpus" --data_type ${data_type} --data ${data}
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  echo "Stage 14: score stage-3 ordinary and uncertainty-aware trials ..."
  local/score.sh --stage 1 --stop-stage 2 --data ${data} \
    --exp_dir ${stage3_eval_exp} --trials "$trials"
  local/score_uncertainty.sh --stage 1 --stop-stage 2 --data ${data} \
    --exp_dir ${stage3_eval_exp} --trials "$trials"
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  echo "Stage 15: AS-norm and uncertainty-aware AS-norm (stage-3) ..."
  # local/score_norm.sh \
  #   --stage 1 --stop-stage 3 \
  #   --score_norm_method $score_norm_method \
  #   --cohort_set vox2_dev \
  #   --top_n $top_n \
  #   --data ${data} \
  #   --exp_dir ${stage3_eval_exp} \
  #   --trials "$trials"

  local/score_norm_uncertainty.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir ${stage3_eval_exp} \
    --trials "$trials"
fi
