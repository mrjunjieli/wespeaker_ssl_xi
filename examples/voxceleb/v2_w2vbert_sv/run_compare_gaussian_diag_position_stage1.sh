#!/bin/bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./path.sh
export CUDA_VISIBLE_DEVICES=2,3
export PYTHON_BIN=${PYTHON_BIN:-python}
exp=exp/v2_w2vbert_sv/gaussian_diag_position_comparison
with_position=exp/v2_w2vbert_sv/gaussian_diag_transformer_pos
without_position=exp/v2_w2vbert_sv/gaussian_diag_transformer_nopos
stage=1
stop_stage=4
prepare_only=false
. tools/parse_options.sh
[[ "$stage" =~ ^[0-9]+$ && "$stop_stage" =~ ^[0-9]+$ ]] &&
  (( stage >= 1 && stop_stage <= 4 && stage <= stop_stage )) || {
  echo 'This comparison only supports Stage1 operations 1-4.' >&2; exit 1;
}
[ "$with_position" != "$without_position" ] || {
  echo 'Comparison requires distinct experiment prefixes.' >&2; exit 1;
}
mkdir -p "$(dirname "$exp")"
exec 8>"${exp}.lock"
flock -n 8 || { echo 'Comparison is already running.' >&2; exit 1; }
state="${exp}.state"
update_state() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$state"; }
trap 'code=$?; update_state "FAILED exit=$code line=$LINENO"; exit "$code"' ERR
for variant in transformer_pos transformer_nopos; do
  if [ "$variant" = transformer_pos ]; then
    directory=$with_position
  else
    directory=$without_position
  fi
  update_state "RUNNING_$variant stage=$stage stop_stage=$stop_stage"
  bash run_gaussian_diag.sh --variant "$variant" --exp "$directory" \
    --stage "$stage" --stop_stage "$stop_stage" --prepare_only "$prepare_only"
done
if [ "$prepare_only" = false ] && [ "$stop_stage" -eq 4 ]; then
  "$PYTHON_BIN" local/summarize_gaussian_diag.py \
    --with-position "${with_position}_stage1" \
    --without-position "${without_position}_stage1" --output "${exp}.tsv"
fi
update_state 'COMPLETED_WAITING_USER_DECISION; no Stage2 or Stage3'
