# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
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

import os
from pathlib import Path

import fire
import kaldiio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch 


def calculate_mean_from_kaldi_vec(scp_path):
    vec_num = 0
    mean_vec = None

    for _, vec in kaldiio.load_scp_sequential(scp_path):
        if mean_vec is None:
            mean_vec = np.zeros_like(vec)
        mean_vec += vec
        vec_num += 1

    return mean_vec / vec_num


def calculate_var_scale_from_scp(var_scp_path,
                                 cache_path=None,
                                 max_samples=200000):
    """Per-dimension median of the training/cohort variance vectors.

    Used as the scale vector c_d in
        alpha = sqrt(|e|^2 / sum(e^2 / (var + rho * c_d)))
    so that the weighting adapts to both the variance magnitude of the
    model (~0.4 for U-Cube XI pooling, ~0.0025 for SSL Gaussian
    posterior) and the per-dimension scale differences (large for XI
    pooling, nearly uniform for the SSL Gaussian posterior).
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)
    chunks = []
    for _, var in kaldiio.load_scp_sequential(var_scp_path):
        chunks.append(np.asarray(var, dtype=np.float32))
    if not chunks:
        return 1.0
    V = np.stack(chunks)
    if len(V) > max_samples:  # median is robust; subsample to bound memory
        sel = np.linspace(0, len(V) - 1, max_samples).astype(np.int64)
        V = V[sel]
    var_c = np.median(V, axis=0)
    if cache_path is not None:
        np.save(cache_path, var_c)
    return var_c


def trials_cosine_score(eval_scp_path='',
                        store_dir='',
                        mean_vec=None,
                        trials=(),
                        var_c=1.0,
                        rho=1.0,
                        gamma=0.5):
    if mean_vec is None or not os.path.exists(mean_vec):
        mean_vec = 0.0
    else:
        mean_vec = np.load(mean_vec)

    # each embedding may be accessed multiple times, here we pre-load them
    # into the memory
    emb_dict = {}
    var_dict = {}
    var_scp_path = os.path.dirname(eval_scp_path)
    var_scp_path = var_scp_path+'/xvector_variance.scp'
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb = emb - mean_vec
        emb_dict[utt] = emb
    
    for utt, var in kaldiio.load_scp_sequential(var_scp_path):
        var_dict[utt] = var

    for trial in trials:
        store_path = os.path.join(store_dir,
                                  os.path.basename(trial) + '.score')
        with open(trial, 'r') as trial_r, open(store_path, 'w') as w_f:
            lines = trial_r.readlines()
            for line in tqdm(lines,
                             desc='scoring trial {}'.format(
                                 os.path.basename(trial))):
                segs = line.strip().split()
                emb1, emb2 = emb_dict[segs[0]], emb_dict[segs[1]]
                var1, var2 = var_dict[segs[0]], var_dict[segs[1]]  #192
                
                alpha_1 = np.dot(emb1, emb1)/np.sum(
                    emb1 / (var1 + rho * var_c) * emb1)
                alpha_1 = alpha_1**0.5

                alpha_2 = np.dot(emb2, emb2)/np.sum(
                    emb2 / (var2 + rho * var_c) * emb2)
                alpha_2 = alpha_2**0.5
                cos_score = cosine_similarity(emb1.reshape(1, -1),
                                              emb2.reshape(1, -1))[0][0]
                # gamma softens the multiplicative weighting (gamma=1 is the
                # full weight); gamma=0.5 keeps the EER gain while avoiding
                # minDCF regression at low-FA operating points
                cos_score = (alpha_1 * alpha_2)**gamma * cos_score
                
                if len(segs) == 3:  # enroll_name test_name target/nontarget
                    w_f.write('{} {} {:.5f} {}\n'.format(
                        segs[0], segs[1], cos_score, segs[2]))
                else:  # enroll_name test_name
                    w_f.write('{} {} {:.5f}\n'.format(segs[0], segs[1],
                                                      cos_score))


def main(exp_dir, eval_scp_path, cal_mean, cal_mean_dir, *trials,
         rho=1.0, gamma=0.5):

    print(cal_mean)
    if not cal_mean:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec_path = None
    else:
        scp_path = os.path.join(cal_mean_dir, 'xvector.scp')
        print("Calculate mean statistics from {}.".format(scp_path))
        mean_vec = calculate_mean_from_kaldi_vec(scp_path)
        mean_vec_path = os.path.join(cal_mean_dir, 'mean_vec.npy')
        np.save(mean_vec_path, mean_vec)

    # scoring trials
    var_c = 1.0
    if cal_mean_dir:
        var_scp = os.path.join(cal_mean_dir, 'xvector_variance.scp')
        if os.path.exists(var_scp):
            var_c = calculate_var_scale_from_scp(
                var_scp, cache_path=os.path.join(cal_mean_dir,
                                                 'var_scale_perdim.npy'))
            print("Uncertainty scale c_d = rho({}) * per-dim median of "
                  "train variance, range [{:.6f}, {:.6f}]".format(
                      rho, np.min(var_c), np.max(var_c)))
    store_score_dir = os.path.join(exp_dir, 'scores_uncertainty')
    Path(store_score_dir).mkdir(parents=True, exist_ok=True)
    trials_cosine_score(eval_scp_path, store_score_dir, mean_vec_path,
                        trials, var_c=var_c, rho=rho, gamma=gamma)


if __name__ == "__main__":
    fire.Fire(main)
