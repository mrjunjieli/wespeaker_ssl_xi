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

"""MHFA-LN-U-Cube backend for InterTopK/Subcenter uncertainty training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MHFA import GradMultiply
from .multi_view_attention import MultiViewTransformerEncoderLayer


class SSL_BACKEND_MHFA_LN_UCube_InterTopKSubcenter(nn.Module):
    """MHFA fusion with shared Value and multi-posterior U-Cube pooling.

    The temporal precision encoder is shared, while each pooling head has an
    independent precision map and Gaussian prior. The Value representation is
    intentionally shared across heads, matching the original MHFA design.
    """

    def __init__(self,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=0.0,
                 uncertainty_hidden_size=256,
                 uncertainty_num_heads=8,
                 uncertainty_pooling_heads=8,
                 uncertainty_dropout=0.2,
                 train_prior_mean=True,
                 train_prior_prec=True):
        super().__init__()
        if uncertainty_pooling_heads < 1:
            raise ValueError("uncertainty_pooling_heads must be positive")
        self.feature_grad_mult = feature_grad_mult
        self.pooling_heads = uncertainty_pooling_heads
        self.compression_dim = compression_dim
        self.weights_k = nn.Parameter(torch.ones(nb_layer))
        self.weights_v = nn.Parameter(torch.ones(nb_layer))

        self.cmp_linear_k = nn.Linear(feat_dim, compression_dim)
        self.cmp_linear_v = nn.Linear(feat_dim, compression_dim)
        self.norm_k = nn.LayerNorm(compression_dim)
        self.norm_v = nn.LayerNorm(compression_dim)

        self.precision_input = nn.Sequential(
            nn.Conv1d(compression_dim,
                      uncertainty_hidden_size,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(uncertainty_hidden_size),
        )
        self.precision_encoder = MultiViewTransformerEncoderLayer(
            embed_dim=uncertainty_hidden_size,
            num_heads=uncertainty_num_heads,
            ff_hidden=uncertainty_hidden_size * 4,
            dropout=uncertainty_dropout,
        )
        self.precision_output = nn.Conv1d(
            uncertainty_hidden_size,
            compression_dim * uncertainty_pooling_heads,
            kernel_size=1)
        self.softplus = nn.Softplus(beta=1, threshold=20)

        self.prior_mean = nn.Parameter(
            torch.zeros(1, uncertainty_pooling_heads, compression_dim),
            requires_grad=train_prior_mean)
        self.prior_logprec = nn.Parameter(
            torch.zeros(1, uncertainty_pooling_heads, compression_dim),
            requires_grad=train_prior_prec)
        self.pooling_fc = nn.Linear(
            compression_dim * uncertainty_pooling_heads, embed_dim)

    def _estimate_log_precision(self, key):
        hidden = self.precision_input(key.transpose(1, 2))
        hidden = self.precision_encoder(hidden.transpose(1, 2))
        precision = self.softplus(
            self.precision_output(hidden.transpose(1, 2))).transpose(1, 2)
        batch, frames, _ = precision.shape
        precision = precision.reshape(
            batch, frames, self.pooling_heads, self.compression_dim)
        log_precision = 2.0 * torch.log(precision.clamp_min(1.0e-8))
        return log_precision.clamp(-15.0, 15.0)

    def _posterior_pooling(self, value, log_precision):
        batch = value.shape[0]
        values = value.unsqueeze(2).expand(
            -1, -1, self.pooling_heads, -1)
        prior_mean = self.prior_mean.expand(batch, -1, -1).unsqueeze(1)
        prior_logprec = self.prior_logprec.expand(batch, -1, -1).unsqueeze(1)
        values = torch.cat((values, prior_mean), dim=1)
        logprec = torch.cat((log_precision, prior_logprec), dim=1)
        weights = torch.softmax(logprec, dim=1)
        posterior_mean = torch.sum(values * weights, dim=1)
        posterior_precision = torch.exp(logprec).sum(dim=1)
        posterior_variance = posterior_precision.clamp_min(1.0e-12).reciprocal()
        return posterior_mean, posterior_variance

    def forward(self, x):
        # x: [batch, feature, frames, layers]
        x = GradMultiply.apply(x, self.feature_grad_mult)
        key = torch.sum(
            x * F.softmax(self.weights_k, dim=-1), dim=-1).transpose(1, 2)
        value = torch.sum(
            x * F.softmax(self.weights_v, dim=-1), dim=-1).transpose(1, 2)
        key = self.norm_k(self.cmp_linear_k(key))
        value = self.norm_v(self.cmp_linear_v(value))

        log_precision = self._estimate_log_precision(key)
        posterior_mean, posterior_variance = self._posterior_pooling(
            value, log_precision)
        posterior_mean = posterior_mean.reshape(
            value.shape[0], self.pooling_heads * self.compression_dim)
        posterior_variance = posterior_variance.reshape(
            value.shape[0], self.pooling_heads * self.compression_dim)
        embedding = self.pooling_fc(posterior_mean)
        embedding_variance = (
            posterior_variance @ self.pooling_fc.weight.square().T)
        return value, embedding_variance, embedding
