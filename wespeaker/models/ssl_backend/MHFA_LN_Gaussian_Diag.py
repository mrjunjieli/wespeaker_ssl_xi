"""Per-head, per-feature diagonal Gaussian log-precision pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_GaussianLogPrec

__all__ = ['SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag']


def diagonal_posterior(value, logits, prior_mean, prior_logprec):
    """Pool [B,T,H,D] log-precisions into [B,H,D] mean and variance."""
    with torch.autocast(device_type=value.device.type, enabled=False):
        value = value.float()
        logits = logits.float().clamp(-15.0, 15.0)
        prior_logprec = prior_logprec.float().clamp(-15.0, 15.0)
        log_normalizer = torch.logaddexp(
            torch.logsumexp(logits, dim=1), prior_logprec)
        weights = torch.exp(logits - log_normalizer.unsqueeze(1))
        mean = torch.einsum('bthd,btd->bhd', weights, value)
        mean = mean + torch.exp(prior_logprec - log_normalizer) * prior_mean.float()
        variance = torch.exp(-log_normalizer)
        return mean, variance


class DiagonalGaussianMixin:
    """Replace scalar precision without changing the parent's feature pathway.

    Chunking and activation recomputation bound the large [B,T,H,D] pooling
    intermediates. The projected embedding retains diagonal-only uncertainty
    propagation, including the existing independence approximation across heads.
    """

    def __init__(self, *args, pooling_head_chunk_size=8,
                 pooling_checkpoint=True, verbose=False, **kwargs):
        if not isinstance(pooling_head_chunk_size, int) or pooling_head_chunk_size < 1:
            raise ValueError('pooling_head_chunk_size must be a positive integer')
        super().__init__(*args, verbose=False, **kwargs)
        self.head_nb = getattr(self, 'head_nb', 1)
        self.pooling_head_chunk_size = pooling_head_chunk_size
        self.pooling_checkpoint = pooling_checkpoint
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb * self.cmp_dim)
        self.prior_logprec = nn.Parameter(
            torch.zeros(1, self.head_nb, self.cmp_dim),
            requires_grad=self.prior_logprec.requires_grad)
        if self.prior_mean.ndim == 2:
            self.prior_mean = nn.Parameter(
                self.prior_mean.detach().unsqueeze(1),
                requires_grad=self.prior_mean.requires_grad)
        if verbose:
            self.print_structure()

    def _pool_inputs(self, x):
        return self._fuse_features(x)

    def _posterior_pooling(self, value, att_logits):
        batch, frames, _ = value.shape
        logits = att_logits.reshape(batch, frames, self.head_nb, self.cmp_dim)
        return diagonal_posterior(
            value, logits, self.prior_mean, self.prior_logprec)

    def _project_and_pool(self, key, value, weight, bias, prior_mean, prior_logprec):
        with torch.autocast(device_type=key.device.type, enabled=False):
            logits = F.linear(key.float(), weight.float(), bias.float())
            logits = logits.reshape(key.shape[0], key.shape[1], -1, self.cmp_dim)
            return diagonal_posterior(value, logits, prior_mean, prior_logprec)

    def _pool_from_key(self, key, value):
        means, variances = [], []
        for start in range(0, self.head_nb, self.pooling_head_chunk_size):
            stop = min(start + self.pooling_head_chunk_size, self.head_nb)
            inputs = (
                key, value,
                self.att_head.weight[start * self.cmp_dim:stop * self.cmp_dim],
                self.att_head.bias[start * self.cmp_dim:stop * self.cmp_dim],
                self.prior_mean[:, start:stop],
                self.prior_logprec[:, start:stop])
            if self.pooling_checkpoint and self.training and torch.is_grad_enabled():
                mean, variance = checkpoint(
                    self._project_and_pool, *inputs, use_reentrant=False)
            else:
                mean, variance = self._project_and_pool(*inputs)
            means.append(mean)
            variances.append(variance)
        return torch.cat(means, dim=1), torch.cat(variances, dim=1)

    def forward(self, x):
        value_key, value = self._pool_inputs(x)
        mean, variance = self._pool_from_key(value_key, value)
        mean = mean.flatten(1)
        variance = variance.flatten(1)
        embedding = self.pooling_fc(self.bn_embed(mean))
        with torch.autocast(device_type=value.device.type, enabled=False):
            variance = variance * self.bn_embed.weight.float().square() / (
                self.bn_embed.running_var.float() + self.bn_embed.eps)
            embedding_variance = F.linear(
                variance, self.pooling_fc.weight.float().square())
        return value, embedding_variance, embedding

    def print_structure(self):
        print(type(self).__name__)
        print(f'Feature pathway: {type(self).__mro__[2].__name__}')
        print(f'Precision: Linear({self.cmp_dim}, {self.head_nb * self.cmp_dim})')
        print(f'Prior and posterior: [B, {self.head_nb}, {self.cmp_dim}]')
        print('Independent precision per head and feature; diagonal covariance')
        print(f'Pooling head chunk size: {self.pooling_head_chunk_size}')
        print(f'Pooling activation checkpointing: {self.pooling_checkpoint}')
        print(f'Embedding: {self.head_nb * self.cmp_dim} -> {self.ous_dim}')
        if hasattr(self, 'temporal_encoder'):
            print(f'Position encoding: {self.transformer_use_position_encoding}')
            print(self.temporal_encoder)


class SSL_BACKEND_MHFA_LN_GaussianLogPrec_Diag(
        DiagonalGaussianMixin, SSL_BACKEND_MHFA_LN_GaussianLogPrec):
    """No-adapter MHFA with per-head, per-feature log-precision."""
