# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
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
"""Simple weighted-sum pooling with Gaussian posterior uncertainty.

Ablation of the MHFA-LN Gaussian recipe: this backend drops both the per-layer
adapter and the multi-head frame attention (MHFA).  Each SSL layer feature is
fused by a single learnable layer weight, compressed, and a single scalar
frame log-precision (one head) drives a Gaussian log-precision posterior
(weighted sum) over frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MHFA import GradMultiply


class SSL_BACKEND_LN_GaussianLogPrec_WeightedSum(nn.Module):
    """LayerNorm + simple weighted-sum pooling with Gaussian log-precision.

    No per-layer adapter, no multi-head attention.  A single path fuses the
    SSL layers, then a scalar (single-head) frame log-precision produces a
    Gaussian posterior mean/variance over frames.
    """

    def __init__(self,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0,
                 train_prior_mean=True,
                 train_prior_prec=True,
                 verbose=False):
        super().__init__()

        self.feature_grad_mult = feature_grad_mult
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim
        self.nb_layer = nb_layer

        # Layer fusion: single learnable weight set (no K/V split).
        self.weights = nn.Parameter(torch.ones(nb_layer))

        # Compression + normalization
        self.cmp_linear = nn.Linear(feat_dim, compression_dim)
        self.norm = nn.LayerNorm(compression_dim)

        # Simple scalar frame attention (single head -> weighted sum).
        self.att_head = nn.Linear(compression_dim, 1)

        # Gaussian prior parameters (single head).
        self.prior_mean = nn.Parameter(
            torch.zeros(1, compression_dim),
            requires_grad=train_prior_mean)
        self.prior_logprec = nn.Parameter(
            torch.zeros(1, 1),
            requires_grad=train_prior_prec)

        # Embedding projection
        self.bn_embed = nn.BatchNorm1d(compression_dim)
        self.pooling_fc = nn.Linear(compression_dim, embed_dim)

        if verbose:
            self.print_structure()

    def print_structure(self):
        print("SSL_BACKEND_LN_GaussianLogPrec_WeightedSum")
        print("├── layer fusion: learnable weights (single path, no K/V)")
        print("├── compression: Linear + LayerNorm")
        print("├── frame attention: Linear(compression_dim → 1) scalar log-precision")
        print("├── Gaussian prior: trainable mean/log precision")
        print("├── posterior: softmax(log-precision) weighted sum")
        print("└── embedding: Linear(compression_dim → embed_dim)")

    def _fuse_features(self, x):
        x = GradMultiply.apply(x, self.feature_grad_mult)
        fused = torch.sum(
            x * F.softmax(self.weights, dim=-1), dim=-1).transpose(1, 2)
        fused = self.norm(self.cmp_linear(fused))
        return fused

    def _posterior_pooling(self, value, att_logits):
        batch, _, _ = value.shape

        frame_logprec = att_logits.transpose(1, 2)  # [B, 1, T]
        prior_logprec = self.prior_logprec.expand(batch, -1).unsqueeze(-1)
        all_logprec = torch.cat((frame_logprec, prior_logprec), dim=2)
        all_logprec = all_logprec.clamp(min=-15.0, max=15.0)

        values = value.unsqueeze(1)  # [B, 1, T, D]
        prior_mean = self.prior_mean.expand(batch, -1, -1).unsqueeze(2)
        all_values = torch.cat((values, prior_mean), dim=2)

        weight_attn = torch.softmax(all_logprec, dim=2)
        posterior_mean = torch.sum(
            all_values * weight_attn.unsqueeze(-1), dim=2)

        posterior_precision = torch.sum(torch.exp(all_logprec), dim=2)
        posterior_variance = posterior_precision.clamp_min(
            1.0e-6).reciprocal().unsqueeze(-1)
        posterior_variance = posterior_variance.expand(
            -1, -1, self.cmp_dim)
        return posterior_mean, posterior_variance

    def forward(self, x):
        # x: [batch, feat_dim, frames, layers]
        value = self._fuse_features(x)  # [B, T, D]
        posterior_mean, posterior_variance = self._posterior_pooling(
            value, self.att_head(value))

        posterior_mean = posterior_mean.reshape(value.shape[0], self.cmp_dim)
        posterior_variance = posterior_variance.reshape(
            value.shape[0], self.cmp_dim)
        posterior_mean = self.bn_embed(posterior_mean)
        posterior_variance = (self.bn_embed.weight ** 2 * posterior_variance
                              / (self.bn_embed.running_var + self.bn_embed.eps))
        embedding = self.pooling_fc(posterior_mean)
        embedding_variance = posterior_variance @ self.pooling_fc.weight.square().T

        return value, embedding_variance, embedding
