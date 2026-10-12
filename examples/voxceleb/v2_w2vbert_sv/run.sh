#!/bin/bash

# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#               2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Extended from the original WeSpeaker/W2V-BERT recipe with MHFA-LN +
# U-Cube pooling, uncertainty curriculum, and a LoRA-to-full-FT handoff.
#
# Two-stage W2V-BERT 2.0 + MHFA-LN-U-Cube speaker verification recipe.
#   stage 1: freeze the W2V-BERT base and train Q/K LoRA adapters together
#            with the MHFA-LN-U-Cube backend and classifier
#            (conf/w2vbert_sv_mhfa_stage1_frozen.yaml)
#   stage 2: average the final stage-1 checkpoints, extract embeddings from
#            the averaged model, and evaluate it on VoxCeleb1-O/E/H
#   stage 3: merge the stage-1 LoRA adapters into W2V-BERT, then unfreeze the
#            complete frontend and run stage-2 full-model fine-tuning
#            (conf/w2vbert_sv_mhfa_stage2_ft.yaml)
#   stage 4: average the final stage-2 full-fine-tuning checkpoints
#   stage 5: use the averaged stage-2 model to extract VoxCeleb2 development
#            embeddings for mean normalization and VoxCeleb1 test embeddings
#   stage 6: score the stage-2 embeddings and report EER/minDCF on
#            VoxCeleb1-O/E/H
#
# Training and extraction audio are read from shard lists. MUSAN noise and
# RIR augmentation data are read independently from their LMDBs.

. ./path.sh || exit 1

stage=2
stop_stage=2

data=data
data_type="shard"

stage1_config=conf/w2vbert_sv_mhfa_stage1_frozen.yaml
stage1_exp=exp/v2_w2vbert_sv/stage1_frozen_vox2
stage1_eval_exp=${stage1_exp}/eval
stage2_config=conf/w2vbert_sv_mhfa_stage2_ft.yaml
stage2_exp=exp/v2_w2vbert_sv/stage2_ft_vox2

gpus="[2,3]"
stage1_num_avg=4
stage2_num_avg=1
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"

. tools/parse_options.sh || exit 1

num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: train with frozen frontend ..."
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $stage1_config \
      --exp_dir ${stage1_exp} \
      --gpus $gpus \
      --num_avg ${stage1_num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: average stage-1 models ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage1_exp}/models/avg_model.pt \
    --src_path ${stage1_exp}/models \
    --num ${stage1_num_avg}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: extract and score stage-1 average ..."
  mkdir -p ${stage1_eval_exp}
  cp ${stage1_exp}/config.yaml ${stage1_eval_exp}/config.yaml
  local/extract_vox.sh \
    --exp_dir ${stage1_eval_exp} \
    --model_path ${stage1_exp}/models/avg_model.pt \
    --nj 8 --gpus $gpus --data_type ${data_type} --data ${data}
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir ${stage1_eval_exp} \
    --trials "$trials"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: merge LoRA and full fine-tuning from stage 1 ..."
  mkdir -p ${stage2_exp}/models
  python wespeaker/bin/merge_w2vbert_lora.py \
    --config ${stage1_exp}/config.yaml \
    --src_model ${stage1_exp}/models/avg_model.pt \
    --dst_model ${stage2_exp}/models/model_0.pt
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $stage2_config \
      --exp_dir ${stage2_exp} \
      --gpus $gpus \
      --num_avg ${stage2_num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --checkpoint ${stage2_exp}/models/model_0.pt
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: average stage-2 models ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage2_exp}/models/avg_model.pt \
    --src_path ${stage2_exp}/models \
    --num ${stage2_num_avg}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: extract vox2_dev + vox1 embeddings ..."
  local/extract_vox.sh \
    --exp_dir ${stage2_exp} \
    --model_path ${stage2_exp}/models/avg_model.pt \
    --nj 8 --gpus $gpus --data_type ${data_type} --data ${data}
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir ${stage2_exp} \
    --trials "$trials"
fi
