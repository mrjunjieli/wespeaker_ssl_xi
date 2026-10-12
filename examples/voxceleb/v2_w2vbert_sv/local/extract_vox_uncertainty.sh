#!/bin/bash
set -euo pipefail

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

exp_dir=''
model_path=''
nj=4
nj_vox2=''  # defaults to ${nj}; vox2_dev is ~7x larger than vox1
gpus="[0,1]"
data_type="shard"  # shard/raw/feat
data=data

WESPEAKER_ROOT=${WESPEAKER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}

. "$WESPEAKER_ROOT/tools/parse_options.sh"

data_name_array=("vox2_dev" "vox1")
data_list_path_array=("${data}/vox2_dev/${data_type}.list" "${data}/vox1/${data_type}.list")
data_scp_path_array=("${data}/vox2_dev/wav.scp" "${data}/vox1/wav.scp") # to count the number of wavs
nj_vox2=${nj_vox2:-$nj}
nj_array=($nj_vox2 $nj)
batch_size_array=(1 1) # full-utterance extraction; batching lacks length masking
num_workers_array=(4 1)
count=${#data_name_array[@]}

# Extract serially: vox1 first (smaller), then vox2_dev (larger)
for i in $(seq 0 $(($count - 1))); do
  wavs_num=$(wc -l < "${data_scp_path_array[$i]}")
  echo "Extracting ${data_name_array[$i]} (${wavs_num} utterances) ..."
  CUDA_VISIBLE_DEVICES=2,3 \
  WESPEAKER_ROOT="$WESPEAKER_ROOT" PYTHON_BIN="${PYTHON_BIN:-python}" \
    bash "$WESPEAKER_ROOT/tools/extract_embedding_uncertainty.sh" --exp_dir "${exp_dir}" \
    --model_path "$model_path" \
    --data_type "${data_type}" \
    --data_list "${data_list_path_array[$i]}" \
    --wavs_num "${wavs_num}" \
    --store_dir "${data_name_array[$i]}" \
    --batch_size "${batch_size_array[$i]}" \
    --num_workers "${num_workers_array[$i]}" \
    --nj "${nj_array[$i]}" \
    --gpus "$gpus"
  # No & — waits for completion before next iteration
  if [ $? -ne 0 ]; then
    echo "Extraction failed for ${data_name_array[$i]}" >&2
    exit 1
  fi
done

echo "Embedding dir is (${exp_dir}/embeddings)."
