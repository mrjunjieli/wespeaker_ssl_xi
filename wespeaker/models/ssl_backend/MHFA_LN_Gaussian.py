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
"""MHFA-LN pooling with Gaussian posterior uncertainty."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MHFA import GradMultiply


class SSL_BACKEND_MHFA_LN_Gaussian(nn.Module):
    """Interpret MHFA frame attention as scalar Gaussian precision.

    The baseline MHFA key/value pathway is unchanged.  Each head's attention
    logit produces a positive frame precision, which is summed over time with
    a trainable per-head prior precision for Gaussian posterior pooling.
    """

    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0,
                 train_prior_mean=True,
                 train_prior_prec=True,
                 verbose=False):
        super().__init__()
        if head_nb < 1:
            raise ValueError("head_nb must be positive")

        self.feature_grad_mult = feature_grad_mult
        self.head_nb = head_nb
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim

        self.weights_k = nn.Parameter(torch.ones(nb_layer))
        self.weights_v = nn.Parameter(torch.ones(nb_layer))
        self.cmp_linear_k = nn.Linear(feat_dim, compression_dim)
        self.cmp_linear_v = nn.Linear(feat_dim, compression_dim)
        self.norm_k = nn.LayerNorm(compression_dim)
        self.norm_v = nn.LayerNorm(compression_dim)
        self.att_head = nn.Linear(compression_dim, head_nb)

        self.prior_mean = nn.Parameter(
            torch.zeros(1, head_nb, compression_dim),
            requires_grad=train_prior_mean)
        self.prior_logprec = nn.Parameter(
            torch.zeros(1, head_nb),
            requires_grad=train_prior_prec)
        self.bn_embed = nn.BatchNorm1d(head_nb * compression_dim)
        self.pooling_fc = nn.Linear(head_nb * compression_dim, embed_dim)

        if verbose:
            self.print_structure()

    def print_structure(self):
        print("SSL_BACKEND_MHFA_LN_Gaussian")
        print("├── layer fusion: learnable K/V weights")
        print("├── K/V compression: Linear + LayerNorm")
        print("├── MHFA attention: Linear(compression_dim → head_nb)")
        print("├── frame precision: Softplus(attention logits), scalar per head")
        print("├── Gaussian prior: trainable per-head mean/log precision")
        print("├── posterior: precision sum and precision-weighted mean")
        print("└── embedding: Linear( head_nb × compression_dim → embed_dim )")

    def _fuse_features(self, x):
        x = GradMultiply.apply(x, self.feature_grad_mult)
        key = torch.sum(
            x * F.softmax(self.weights_k, dim=-1), dim=-1).transpose(1, 2)
        value = torch.sum(
            x * F.softmax(self.weights_v, dim=-1), dim=-1).transpose(1, 2)
        key = self.norm_k(self.cmp_linear_k(key))
        value = self.norm_v(self.cmp_linear_v(value))
        return key, value

    def _posterior_pooling(self, value, att_logits):
        batch, _, _ = value.shape
        # Softplus keeps precision positive without imposing a fixed sum over
        # frames (unlike softmax attention).  The scalar is shared by values
        # within one MHFA head, matching the original attention semantics.
        frame_precision = F.softplus(att_logits) + 1.0e-6  # [B, T, H]
        prior_precision = torch.exp(self.prior_logprec.clamp(-15.0, 15.0))
        prior_precision = prior_precision.expand(batch, -1)  # [B, H]
        posterior_precision = prior_precision + frame_precision.sum(dim=1)

        values = value.unsqueeze(2).expand(
            -1, -1, self.head_nb, -1)  # [B, T, H, D]
        weighted_values = (
            values * frame_precision.unsqueeze(-1)).sum(dim=1)
        prior_mean = self.prior_mean.expand(batch, -1, -1)
        posterior_mean = (
            weighted_values + prior_precision.unsqueeze(-1) * prior_mean
        ) / posterior_precision.unsqueeze(-1).clamp_min(1.0e-6)
        posterior_variance = posterior_precision.reciprocal().unsqueeze(-1)
        posterior_variance = posterior_variance.expand(-1, -1, self.cmp_dim)
        return posterior_mean, posterior_variance

    def forward(self, x):
        # x: [batch, feature, frames, layers]
        key, value = self._fuse_features(x)
        posterior_mean, posterior_variance = self._posterior_pooling(
            value, self.att_head(key))
        posterior_mean = posterior_mean.reshape(
            value.shape[0], self.head_nb * self.cmp_dim)
        posterior_variance = posterior_variance.reshape(
            value.shape[0], self.head_nb * self.cmp_dim)
        posterior_mean = self.bn_embed(posterior_mean)
        posterior_variance = (self.bn_embed.weight ** 2 * posterior_variance
                              / (self.bn_embed.running_var + self.bn_embed.eps))
        embedding = self.pooling_fc(posterior_mean)
        embedding_variance = posterior_variance @ self.pooling_fc.weight.square().T
        return value, embedding_variance, embedding


class SSL_BACKEND_MHFA_LN_GaussianLogPrec(SSL_BACKEND_MHFA_LN_Gaussian):
    """Gaussian MHFA using MHFA logits as frame log-precisions.

    Unlike the original Gaussian class above, this version follows the
    established Gaussian posterior implementation exactly: attention logits
    are interpreted directly as log-precision, while softmax over frame and
    prior log-precisions provides posterior-mean weights.
    """

    def print_structure(self):
        print("SSL_BACKEND_MHFA_LN_GaussianLogPrec")
        print("├── layer fusion: learnable K/V weights")
        print("├── K/V compression: Linear + LayerNorm")
        print("├── MHFA attention logits = frame log-precision")
        print("├── Gaussian prior: trainable per-head mean/log precision")
        print("├── posterior: softmax(log-precision) weighted mean")
        print("└── embedding: Linear( head_nb × compression_dim → embed_dim )")

    def _posterior_pooling(self, value, att_logits):
        batch, _, _ = value.shape

        # MHFA logits are log-precisions.  Clamping matches the previous
        # Gaussian posterior implementation and prevents exp overflow.
        frame_logprec = att_logits.transpose(1, 2)  # [B, H, T]
        prior_logprec = self.prior_logprec.expand(batch, -1).unsqueeze(-1)
        all_logprec = torch.cat((frame_logprec, prior_logprec), dim=2)
        all_logprec = all_logprec.clamp(min=-15.0, max=15.0)

        values = value.unsqueeze(1).expand(
            -1, self.head_nb, -1, -1)  # [B, H, T, D]
        prior_mean = self.prior_mean.expand(batch, -1, -1).unsqueeze(2)
        all_values = torch.cat((values, prior_mean), dim=2)

        # exp(logprec) / sum(exp(logprec)) is exactly precision / posterior
        # precision, so this is the Gaussian posterior mean weight.
        weight_attn = torch.softmax(all_logprec, dim=2)
        posterior_mean = torch.sum(
            all_values * weight_attn.unsqueeze(-1), dim=2)

        posterior_precision = torch.sum(torch.exp(all_logprec), dim=2)
        posterior_variance = posterior_precision.clamp_min(
            1.0e-6).reciprocal().unsqueeze(-1)
        posterior_variance = posterior_variance.expand(
            -1, -1, self.cmp_dim)
        return posterior_mean, posterior_variance
