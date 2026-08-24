# Copyright (c) 2026 Junjie Li (junjie98.li@connect.polyu.hk)
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from torch import nn


class MultiViewSelfAttention(nn.Module):
    """Self-attention whose heads use progressively wider temporal windows."""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 max_window_cap=300):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_window_cap = max_window_cap
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, frames, embed_dim = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch, frames, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q * self.scale, k.transpose(-2, -1))
        indices = torch.arange(frames, device=x.device)
        distance = (indices[:, None] - indices[None, :]).abs()
        radii = torch.tensor(
            [min(2**(i + 1), self.max_window_cap)
             for i in range(self.num_heads)],
            device=x.device)
        allowed = distance.unsqueeze(0) <= radii[:, None, None]
        mask_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~allowed.unsqueeze(0), mask_value)
        if mask is not None:
            logits = logits.masked_fill(
                (~mask).unsqueeze(1).unsqueeze(2), mask_value)

        attention = self.attn_dropout(torch.softmax(logits, dim=-1))
        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch, frames, embed_dim)
        return self.out_dropout(self.out_proj(output))


class MultiViewTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_hidden=2048,
                 dropout=0.2):
        super().__init__()
        self.self_attn = MultiViewSelfAttention(embed_dim,
                                                num_heads,
                                                dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, mask=mask))
        return self.norm2(x + self.ff(x))
