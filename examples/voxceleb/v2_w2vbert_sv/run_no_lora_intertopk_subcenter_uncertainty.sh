#!/bin/bash

# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#               2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Frozen W2V-BERT 2.0 + MHFA-LN-U-Cube speaker verification recipe.
# The backend uses eight MultiView temporal attention heads and eight
# precision/Gaussian posterior pooling heads with one shared Value stream.
# The classifier is uncertainty-aware InterTopK/Subcenter ArcMargin (K=3).
#
#   stage 1: train the backend and classifier with a frozen
#   stage 2: average the final stage-1 checkpoints
#   stage 3: extract VoxCeleb embeddings from the averaged model
#   stage 4: score cleaned VoxCeleb1-O/E/H trials

. ./path.sh || exit 1

stage=1
stop_stage=1

data=data
data_type="shard"

config=conf/w2vbert_sv_mhfa_stage1_frozen_no_lora_intertopk_subcenter_uncertainty.yaml
exp_dir=exp/v2_w2vbert_sv/stage1_frozen_no_lora_intertopk_subcenter_uncertainty_vox2
eval_exp=${exp_dir}/eval

gpus="[2,3]"
num_avg=4
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"

. tools/parse_options.sh || exit 1

num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: train  uncertainty model with frozen frontend ..."
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
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
    --dst_model ${exp_dir}/models/avg_model.pt \
    --src_path ${exp_dir}/models \
    --num ${num_avg}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: extract embeddings from the averaged model ..."
  mkdir -p ${eval_exp}
  cp ${exp_dir}/config.yaml ${eval_exp}/config.yaml
  local/extract_vox.sh \
    --exp_dir ${eval_exp} \
    --model_path ${exp_dir}/models/avg_model.pt \
    --nj 8 --gpus $gpus --data_type ${data_type} --data ${data}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: score cleaned VoxCeleb1-O/E/H trials ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir ${eval_exp} \
    --trials "$trials"
fi
