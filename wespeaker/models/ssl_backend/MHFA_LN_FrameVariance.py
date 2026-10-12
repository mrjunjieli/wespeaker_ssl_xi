# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MHFA-LN with a variance sidecar for uncertainty estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .MHFA import GradMultiply


class SSL_BACKEND_MHFA_LN_FrameVariance(nn.Module):
    """Baseline MHFA-LN mean path plus frame-level value variance.

    The MHFA attention remains a relative frame-importance weight and is not
    interpreted as precision.  The variance sidecar estimates the variance of
    the value observations; the same MHFA weights aggregate it to the variance
    of the utterance representation.
    """

    def __init__(self,
                 head_nb=8,
                 feat_dim=768,
                 compression_dim=128,
                 embed_dim=256,
                 nb_layer=13,
                 feature_grad_mult=1.0,
                 variance_hidden_size=256,
                 variance_dropout=0.0,
                 variance_eps=1.0e-6):
        super(SSL_BACKEND_MHFA_LN_FrameVariance, self).__init__()

        self.feature_grad_mult = feature_grad_mult
        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim
        self.variance_eps = variance_eps

        # Mean path: kept identical to SSL_BACKEND_MHFA_LayerNorm.
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer),
                                      requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer),
                                      requires_grad=True)
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.norm_k = nn.LayerNorm(self.cmp_dim)
        self.norm_v = nn.LayerNorm(self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim,
                                    self.ous_dim)

        # Sidecar: estimate observation variance in the value space.  It does
        # not feed back into the mean path.
        self.variance_head = nn.Sequential(
            nn.Linear(2 * self.cmp_dim, variance_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(variance_dropout),
            nn.Linear(variance_hidden_size,
                      self.head_nb * self.cmp_dim),
        )
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

    def _fuse_features(self, x):
        x = GradMultiply.apply(x, self.feature_grad_mult)
        k = torch.sum(x.mul(F.softmax(self.weights_k, dim=-1)),
                      dim=-1).transpose(1, 2)
        v = torch.sum(x.mul(F.softmax(self.weights_v, dim=-1)),
                      dim=-1).transpose(1, 2)
        k = self.norm_k(self.cmp_linear_k(k))
        v = self.norm_v(self.cmp_linear_v(v))
        return k, v

    def _mean_pooling(self, value, att_logits):
        attention = F.softmax(att_logits, dim=1)
        att_out = value.unsqueeze(2) * attention.unsqueeze(-1)
        return att_out.sum(dim=1), attention

    def _variance_pooling(self, key, value, attention):
        batch, frames, _ = value.shape
        sidecar_input = torch.cat((key, value), dim=-1)
        frame_variance = self.softplus(self.variance_head(sidecar_input))
        frame_variance = frame_variance.reshape(
            batch, frames, self.head_nb, self.cmp_dim)
        frame_variance = frame_variance + self.variance_eps

        pooled_variance = torch.sum(
            attention.square().unsqueeze(-1) * frame_variance, dim=1)
        return pooled_variance

    def get_frame_att_emb(self, x):
        key, value = self._fuse_features(x)
        att_logits = self.att_head(key)
        att_out = value.unsqueeze(-2) * F.softmax(
            att_logits, dim=1).unsqueeze(-1)
        return att_out

    def get_frame_emb(self, x):
        att_out = self.get_frame_att_emb(x)
        return att_out.mean(dim=2)

    def forward(self, x):
        # Input x: [batch, feature, frames, layers].
        key, value = self._fuse_features(x)
        att_logits = self.att_head(key)  # [B, T, H]
        pooling_outs, attention = self._mean_pooling(value, att_logits)
        pooled_variance = self._variance_pooling(key, value, attention)

        # Keep the same [head, channel] flattening as baseline MHFA-LN.
        pooling_outs = pooling_outs.reshape(
            pooling_outs.shape[0], self.head_nb * self.cmp_dim)
        pooled_variance = pooled_variance.reshape(
            pooled_variance.shape[0], self.head_nb * self.cmp_dim)

        embedding = self.pooling_fc(pooling_outs)
        embedding_variance = pooled_variance @ self.pooling_fc.weight.square().T

        # Match the uncertainty executor contract.
        return value, embedding_variance.clamp_min(self.variance_eps), embedding
