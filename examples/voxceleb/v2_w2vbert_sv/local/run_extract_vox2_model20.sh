#!/bin/bash
# Full vox2_dev extraction with model_20 on GPU 0,1.
# Saves embedding + variance arks; prints scale distribution.
set -euo pipefail

BASE=$(cd "$(dirname "$0")/.." && pwd)
WESPEAKER_ROOT=$(cd "$BASE/../../../.." && pwd)
PYTHON=${PYTHON_BIN:-python}

MODEL_PATH="$BASE/exp/v2_w2vbert_sv/mhfa_gaussian/models/model_20.pt"
CONFIG="$BASE/exp/v2_w2vbert_sv/mhfa_gaussian/config.yaml"
DATA_LIST="$BASE/data/vox2_dev/shard.list"
UTT2SPK="$BASE/data/vox2_dev/utt2spk"
OUT_DIR="$BASE/exp/v2_w2vbert_sv/mhfa_gaussian/embeddings/vox2_dev_model20"
LOG_DIR="$OUT_DIR/log"
GPUS=(0 1)
NJ=8
BATCH_SIZE=32
NUM_WORKERS=4
CROP_SEC=2.0

mkdir -p "$LOG_DIR"

# Clean previous outputs
rm -f "$LOG_DIR"/split_* "$OUT_DIR"/xvector_*.ark "$OUT_DIR"/xvector_*.scp \
       "$OUT_DIR"/xvector_*_variance.ark "$OUT_DIR"/xvector_*_variance.scp \
       "$OUT_DIR"/xvector_*_scale_stats.npz

# Split shard.list
data_num=$(wc -l < "$DATA_LIST")
jobs=$(( NJ < data_num ? NJ : data_num ))
subfile_num=$(( (data_num + jobs - 1) / jobs ))
split -l "$subfile_num" -d -a 3 "$DATA_LIST" "$LOG_DIR/split_"
num_gpus=${#GPUS[@]}

echo "Extracting $data_num shards with nj=$jobs across GPU ${GPUS[*]} ..."

pids=()
for suffix in $(seq 0 $((jobs - 1))); do
  idx=$((suffix % num_gpus))
  sfx=$(printf '%03d' "$suffix")
  sub_list="$LOG_DIR/split_${sfx}"
  embed_ark="$OUT_DIR/xvector_${sfx}.ark"
  CUDA_VISIBLE_DEVICES=${GPUS[$idx]} PYTHONPATH="$WESPEAKER_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -u "$BASE/local/extract_vox2_scale.py" \
      --config    "$CONFIG" \
      --model_path "$MODEL_PATH" \
      --data_list  "$sub_list" \
      --embed_ark  "$embed_ark" \
      --utt2spk    "$UTT2SPK" \
      --batch_size "$BATCH_SIZE" \
      --crop_sec   "$CROP_SEC" \
      --num_workers "$NUM_WORKERS" \
    > "$LOG_DIR/split_${sfx}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status != 0 )) && { echo "One or more jobs failed — check $LOG_DIR/*.log" >&2; exit 1; }

# Merge scp files. NOTE: use the explicit suffix list, not the glob
# xvector_*.scp — that glob also matches xvector_000_variance.scp and
# would duplicate every entry in xvector.scp.
merge() {
  local out="$1"; shift
  : > "$out"
  for i in $(seq 0 $((jobs - 1))); do
    local f
    f=$(printf "$1" "$(printf '%03d' "$i")")
    [ -f "$f" ] && cat "$f" >> "$out"
  done
}
merge "$OUT_DIR/xvector.scp"          "$OUT_DIR/xvector_%s.scp"
merge "$OUT_DIR/xvector_variance.scp" "$OUT_DIR/xvector_%s_variance.scp"
echo "Merged → $OUT_DIR/xvector.scp  ($( wc -l < "$OUT_DIR/xvector.scp" ) entries)"
echo "Merged → $OUT_DIR/xvector_variance.scp  ($( wc -l < "$OUT_DIR/xvector_variance.scp" ) entries)"

# Merge and print scale stats
"$PYTHON" - "$OUT_DIR" <<'PYEOF'
import sys, os, glob
import numpy as np

out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, 'xvector_*_scale_stats.npz')))
if not files:
    print("No scale stats files found.")
    sys.exit(0)

merged = {}
for f in files:
    d = np.load(f)
    for k in d.files:
        merged.setdefault(k, []).append(d[k])
merged = {k: np.concatenate(v) for k, v in merged.items()}

np.savez(os.path.join(out_dir, 'scale_stats_merged.npz'), **merged)

def row(name, a):
    return (f"  {name:<22}"
            f"{a.min():>9.4f}{np.percentile(a,25):>9.4f}"
            f"{np.median(a):>9.4f}{np.percentile(a,75):>9.4f}"
            f"{np.percentile(a,95):>9.4f}{a.max():>9.4f}"
            f"{a.mean():>9.4f}{a.std():>9.4f}")

hdr = f"{'':>24}{'min':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p95':>9}{'max':>9}{'mean':>9}{'std':>9}"
n = len(merged['eff_scale'])
print(f"\n{'='*90}")
print(f"  model_20  full vox2_dev  n={n:,}")
print(hdr)
order = ['cosine_gap','Lambda','covariance_mean','intra','unc_scale','eff_scale']
for k in order:
    if k in merged:
        print(row(k, merged[k]))
PYEOF

echo "Done. Scale stats → $OUT_DIR/scale_stats_merged.npz"
