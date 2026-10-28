#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./path.sh

variant=noadapter
stage=1
stop_stage=4
exp=
data=data
data_type=shard
gpus='[2,3]'
checkpoint=
pooling_head_chunk_size=8
nj=2
nj_vox2=2
prepare_only=false
trials='vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi'
. tools/parse_options.sh

case "$variant" in
  noadapter|transformer_pos|transformer_nopos|lightadapter|adapter|weightedsum) ;;
  *) echo "Unknown variant: $variant" >&2; exit 1 ;;
esac
case "$gpus" in
  '[2,3]') export CUDA_VISIBLE_DEVICES=2,3; train_gpus='[0,1]'; num_gpus=2 ;;
  '[2]') export CUDA_VISIBLE_DEVICES=2; train_gpus='[0]'; num_gpus=1 ;;
  '[3]') export CUDA_VISIBLE_DEVICES=3; train_gpus='[0]'; num_gpus=1 ;;
  *) echo 'Only physical GPUs 2 and/or 3 are allowed.' >&2; exit 1 ;;
esac
[[ "$stage" =~ ^[0-9]+$ && "$stop_stage" =~ ^[0-9]+$ ]] &&
  (( stage >= 1 && stop_stage <= 15 && stage <= stop_stage )) || {
  echo 'Require 1 <= stage <= stop_stage <= 15.' >&2; exit 1;
}
export PYTHON_BIN=${PYTHON_BIN:-python}
exp=${exp:-exp/v2_w2vbert_sv/gaussian_diag_${variant}}
mkdir -p "$(dirname "$exp")"
exec 9>"${exp}.lock"
flock -n 9 || { echo "Experiment already running: $exp" >&2; exit 1; }
configs="${exp}_configs"
"$PYTHON_BIN" local/prepare_gaussian_diag_config.py --variant "$variant" \
  --exp-prefix "$exp" --output-dir "$configs" \
  --pooling-head-chunk-size "$pooling_head_chunk_size"
if [ "$prepare_only" = true ]; then
  echo "Configs prepared; no training or GPU initialization: $configs"
  exit 0
fi
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
  nvidia-smi -i "$gpu" --query-gpu=index,memory.used --format=csv
done
"$PYTHON_BIN" -c 'import os, torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))'
state="${exp}.state"
update_state() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$state"; }
trap 'code=$?; update_state "FAILED exit=$code line=$LINENO"; exit "$code"' ERR

train_model() {
  local name=$1 source_checkpoint=$2
  local directory="${exp}_${name}"
  if [ -e "$directory/models/final_model.pt" ]; then
    echo "Already completed: $directory; refusing to overwrite." >&2
    return 1
  fi
  if [ -d "$directory/models" ] && [ -z "$source_checkpoint" ]; then
    echo "Existing models directory requires an explicit diagonal checkpoint: $directory" >&2
    return 1
  fi
  local extra=()
  if [ -n "$source_checkpoint" ]; then
    test -s "$source_checkpoint"
    "$PYTHON_BIN" - "$source_checkpoint" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state = state.get('state_dict', state)
prior = state.get('prior_logprec')
if prior is None or prior.ndim != 3:
    raise SystemExit('Require a diagonal-Gaussian checkpoint; scalar checkpoints are incompatible.')
PY
    extra=(--checkpoint "$source_checkpoint")
  fi
  local average=1
  [ "$name" != stage1 ] || average=4
  update_state "TRAINING_$name variant=$variant"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$num_gpus" wespeaker/bin/train.py \
    --config "$configs/$name.yaml" --exp_dir "$directory" \
    --gpus "$train_gpus" --num_avg "$average" --data_type "$data_type" \
    --train_data "$data/vox2_dev/$data_type.list" \
    --train_label "$data/vox2_dev/utt2spk" \
    --reverb_data "$data/rirs/lmdb" --noise_data "$data/musan/lmdb" "${extra[@]}"
  test -e "$directory/models/final_model.pt"
}
average_model() {
  local directory="${exp}_$1" count=$2
  test -e "$directory/models/final_model.pt"
  update_state "AVERAGING_$1"
  "$PYTHON_BIN" wespeaker/bin/average_model.py \
    --src_path "$directory/models" --dst_model "$directory/models/avg_model.pt" --num "$count"
}
extract_model() {
  local directory="${exp}_$1"
  test -s "$directory/models/avg_model.pt"
  mkdir -p "$directory/eval"
  cp "$directory/config.yaml" "$directory/eval/config.yaml"
  update_state "EXTRACTING_$1"
  bash local/extract_vox_uncertainty.sh --exp_dir "$directory/eval" \
    --model_path "$directory/models/avg_model.pt" --nj "$nj" --nj_vox2 "$nj_vox2" \
    --gpus "$gpus" --data_type "$data_type" --data "$data"
}
score_model() {
  local evaluation="${exp}_$1/eval"
  update_state "SCORING_$1"
  bash -e -o pipefail local/score.sh --stage 1 --stop_stage 2 \
    --data "$data" --exp_dir "$evaluation" --trials "$trials"
  bash -e -o pipefail local/score_uncertainty.sh --stage 1 --stop_stage 2 \
    --data "$data" --exp_dir "$evaluation" --trials "$trials"
}
initialize_model() {
  local source=$1 destination=$2
  test -s "${exp}_${source}/models/avg_model.pt"
  test ! -e "${exp}_${destination}/models/model_0.pt"
  mkdir -p "${exp}_${destination}/models"
  cp "${exp}_${source}/models/avg_model.pt" "${exp}_${destination}/models/model_0.pt"
}
for current in $(seq "$stage" "$stop_stage"); do
  case "$current" in
    1) train_model stage1 "$checkpoint" ;;
    2) average_model stage1 4 ;;
    3) extract_model stage1 ;;
    4) score_model stage1 ;;
    5) initialize_model stage1 stage2_joint ;;
    6) train_model stage2_joint "${exp}_stage2_joint/models/model_0.pt" ;;
    7) average_model stage2_joint 1 ;;
    8) extract_model stage2_joint ;;
    9) score_model stage2_joint ;;
    10) initialize_model stage2_joint stage3_lmft ;;
    11) train_model stage3_lmft "${exp}_stage3_lmft/models/model_0.pt" ;;
    12) average_model stage3_lmft 1 ;;
    13) extract_model stage3_lmft ;;
    14) score_model stage3_lmft ;;
    15) bash -e -o pipefail local/score_norm_uncertainty.sh --stage 1 --stop_stage 3 \
          --score_norm_method asnorm --cohort_set vox2_dev --top_n 300 \
          --data "$data" --exp_dir "${exp}_stage3_lmft/eval" --trials "$trials" ;;
  esac
done
update_state "COMPLETED stop_stage=$stop_stage; waiting for user decision"
