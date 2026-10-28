#!/bin/bash
echo 'DEPRECATED: position-encoding comparison for scalar-per-head variance models only.' >&2
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=2,3
. ./path.sh

wait_pid=${1:?Pass the running positional Stage1 torchrun PID, resume, or restart-nopos}
root=exp/v2_w2vbert_sv
with_position=${root}/mhfa_ln_gaussian_noadapter_transformer_stage1
without_position=${root}/mhfa_ln_gaussian_noadapter_transformer_nopos_stage1
state=${root}/transformer_position_stage1_comparison.state
summary=${root}/transformer_position_stage1_comparison.tsv
mkdir -p "$root"
exec 9>"${root}/transformer_position_stage1_comparison.lock"
flock -n 9 || { echo 'Comparison is already running.' >&2; exit 1; }
update_state() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee "$state"
}
trap 'code=$?; update_state "FAILED exit=$code line=$LINENO"; exit "$code"' ERR

if [ "$wait_pid" = resume ]; then
  checkpoint="$with_position/models/model_29.pt"
  [ -s "$checkpoint" ] && [ ! -e "$with_position/models/final_model.pt" ]
  update_state "RESUMING_POSITIONAL_STAGE1 checkpoint=$checkpoint; weights-only resume, optimizer reset"
  bash deprecated_run_ablation_noadapter_mhfa_transformer.sh \
    --stage 1 --stop_stage 1 --gpus '[2,3]' --checkpoint "$checkpoint"
elif [ "$wait_pid" = restart-nopos ]; then
  update_state 'RESTARTING_NO_POSITION_STAGE1; preserving completed positional Stage1'
else
update_state "WAITING_POSITIONAL_STAGE1 pid=$wait_pid"
if [ -r "/proc/$wait_pid/cmdline" ]; then
  command_line=$(tr '\0' ' ' < "/proc/$wait_pid/cmdline")
  [[ "$command_line" == *torch.distributed.run* && "$command_line" == *mhfa_gaussian_logprec_noadapter_transformer_stage1.yaml* ]] || {
    echo 'Unexpected process; refusing to wait on a different job.' >&2
    exit 1
  }
fi
while kill -0 "$wait_pid" 2>/dev/null; do
  process_state=$(ps -o stat= -p "$wait_pid" || true)
  [[ -z "$process_state" || "$process_state" == Z* ]] && break
  sleep 30
done
fi
[ -s "$with_position/models/model_30.pt" ] && [ -e "$with_position/models/final_model.pt" ] || {
  update_state 'FAILED positional Stage1 did not finish all 30 epochs'
  exit 1
}
for gpu in 2 3; do
  memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( memory > 1500 )); then
    nvidia-smi -i "$gpu"
    update_state "FAILED GPU $gpu occupied: $memory MiB; not switching GPUs"
    exit 1
  fi
done
if [ -d "$without_position/models" ] && find "$without_position/models" -name 'model_*.pt' -print -quit | grep -q .; then
  update_state 'FAILED no-position experiment already has checkpoints; refusing overwrite'
  exit 1
fi

update_state 'TRAINING_NO_POSITION_STAGE1'
bash deprecated_run_ablation_noadapter_mhfa_transformer.sh \
  --stage 1 --stop_stage 1 --gpus '[2,3]' \
  --stage1_config conf/mhfa_gaussian_logprec_noadapter_transformer_nopos_stage1.yaml \
  --stage1_exp "$without_position"
[ -s "$without_position/models/model_30.pt" ] && [ -e "$without_position/models/final_model.pt" ]

for variant in with_position without_position; do
  if [ "$variant" = with_position ]; then
    experiment=$with_position
  else
    experiment=$without_position
  fi
  update_state "EVALUATING_$variant"
  evaluation=$experiment/eval
  python wespeaker/bin/average_model.py \
    --dst_model "$experiment/models/avg_model.pt" \
    --src_path "$experiment/models" --num 4
  mkdir -p "$evaluation"
  cp "$experiment/config.yaml" "$evaluation/config.yaml"
  local/extract_vox_uncertainty.sh \
    --exp_dir "$evaluation" --model_path "$experiment/models/avg_model.pt" \
    --nj 2 --nj_vox2 2 --gpus '[2,3]' --data_type shard --data data
  bash -e -o pipefail local/score.sh \
    --stage 1 --stop_stage 2 --data data --exp_dir "$evaluation"
  bash -e -o pipefail local/score_uncertainty.sh \
    --stage 1 --stop_stage 2 --data data --exp_dir "$evaluation"
done

python - "$with_position" "$without_position" "$summary" <<'PY'
import csv
import re
import sys
from pathlib import Path

roots = dict(zip(('with_position', 'without_position'), map(Path, sys.argv[1:3])))
trials = [f'vox1_{name}_cleaned.kaldi.score' for name in ('O', 'E', 'H')]
pattern = re.compile(
    r'----\s+(\S+)\s+-----\s*EER\s*=\s*([0-9.]+)\s*'
    r'minDCF[^\n]*=\s*([0-9.]+)')
rows = []
metrics = {}
for variant, root in roots.items():
    for scoring, directory in (('cosine', 'scores'), ('uncertainty', 'scores_uncertainty')):
        text = (root / 'eval' / directory / 'vox1_cos_result').read_text()
        results = {trial: (float(eer), float(dcf)) for trial, eer, dcf in pattern.findall(text)}
        for trial in trials:
            eer, dcf = results[trial]
            metrics[variant, scoring, trial] = (eer, dcf)
            rows.append((variant, scoring, trial, eer, dcf))
output = Path(sys.argv[3])
with output.open('w') as stream:
    writer = csv.writer(stream, delimiter='\t')
    writer.writerow(('variant', 'scoring', 'trial', 'EER_percent', 'minDCF'))
    writer.writerows(rows)
print(output.read_text())
print('Delta = without_position - with_position; negative is better for no-position.')
for scoring in ('cosine', 'uncertainty'):
    for trial in trials:
        baseline = metrics['with_position', scoring, trial]
        candidate = metrics['without_position', scoring, trial]
        print(f'{scoring} {trial}: EER delta={candidate[0]-baseline[0]:+.3f}, '
              f'minDCF delta={candidate[1]-baseline[1]:+.3f}')
PY
update_state 'COMPLETED_WAITING_USER_DECISION; no Stage2 or Stage3 will run'
