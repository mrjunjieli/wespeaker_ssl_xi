"""Temporal Transformer before MHFA Gaussian log-precision pooling."""

import math

import torch
import torch.nn as nn

from .MHFA_LN_Gaussian import SSL_BACKEND_MHFA_LN_GaussianLogPrec

__all__ = ['SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer']


class SSL_BACKEND_MHFA_LN_GaussianLogPrec_Transformer(
        SSL_BACKEND_MHFA_LN_GaussianLogPrec):
    """Contextualize compressed K/V sequences with a shared temporal encoder.

    Inputs and outputs follow the Gaussian MHFA backend contract. Attention
    runs over frames, independently for each K/V branch. Fixed sinusoidal
    positions support variable-length, unpadded utterances at extraction.
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
                 transformer_num_layers=1,
                 transformer_nhead=8,
                 transformer_dim_feedforward=512,
                 transformer_dropout=0.1,
                 verbose=False,
                 transformer_use_position_encoding=True):
        if compression_dim < 1:
            raise ValueError('compression_dim must be positive')
        if transformer_nhead < 1 or compression_dim % transformer_nhead:
            raise ValueError('transformer_nhead must divide compression_dim')
        if transformer_num_layers < 1:
            raise ValueError('transformer_num_layers must be positive')
        if transformer_dim_feedforward < 1:
            raise ValueError('transformer_dim_feedforward must be positive')
        if not 0.0 <= transformer_dropout < 1.0:
            raise ValueError('transformer_dropout must be in [0, 1)')
        super().__init__(
            head_nb=head_nb,
            feat_dim=feat_dim,
            compression_dim=compression_dim,
            embed_dim=embed_dim,
            nb_layer=nb_layer,
            feature_grad_mult=feature_grad_mult,
            train_prior_mean=train_prior_mean,
            train_prior_prec=train_prior_prec,
            verbose=False)
        layer = nn.TransformerEncoderLayer(
            d_model=compression_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_num_layers,
            norm=nn.LayerNorm(compression_dim),
            enable_nested_tensor=False)
        self.transformer_use_position_encoding = transformer_use_position_encoding
        self.position_dropout = nn.Dropout(transformer_dropout)
        for encoder_layer in self.temporal_encoder.layers:
            nn.init.xavier_uniform_(encoder_layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(encoder_layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(encoder_layer.linear1.weight)
            nn.init.xavier_uniform_(encoder_layer.linear2.weight)
        if verbose:
            self.print_structure()

    def _position_encoding(self, sequence):
        frames, dim = sequence.shape[1:]
        positions = torch.arange(
            frames, device=sequence.device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, device=sequence.device,
                         dtype=torch.float32) * (-math.log(10000.0) / dim))
        angles = positions * frequencies
        encoding = torch.zeros(
            frames, dim, device=sequence.device, dtype=torch.float32)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :dim // 2].cos()
        return encoding.to(dtype=sequence.dtype).unsqueeze(0)

    def _fuse_features(self, x):
        key, value = super()._fuse_features(x)
        sequences = torch.cat((key, value), dim=0)
        if self.transformer_use_position_encoding:
            sequences = sequences + self._position_encoding(key)
        sequences = self.temporal_encoder(self.position_dropout(sequences))
        return sequences.chunk(2, dim=0)

    def _posterior_pooling(self, value, att_logits):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return super()._posterior_pooling(value.float(), att_logits.float())

    def print_structure(self):
        super().print_structure()
        positions = ('sinusoidal positions + '
                     if self.transformer_use_position_encoding else 'no positions + ')
        print(f'Temporal context: {positions}shared K/V Transformer')
        print(self.temporal_encoder)
