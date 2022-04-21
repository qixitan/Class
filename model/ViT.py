# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: ViT.py
# @Time: 2022/4/20 16:22

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Reduce
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, inplanes: int = 3, patch_size: int = 16, emb_size: int = 768, image_size:int=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(inplanes, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((image_size//patch_size)**2+1, emb_size))

    def forward(self, x):
        out = self.projection(x)
        c, b, w, h = out.shape
        out = out.view(c, b, w*h).permute(0, 2, 1)
        cls_token = self.cls_token.repeat(c, 1, 1)
        out = torch.cat([cls_token, out], dim=1)
        out += self.positions
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 12, dropout: float = 0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # To speed up the calculation, we can use a single matrix to calculate Queries, keys, values.
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):  # 直接继承nn.Sequential 避免重复写forward
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 12,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, num_heads: int = 8, **kwargs):
        super().__init__(*[TransformerEncoderBlock(num_heads=num_heads, **kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class ViT_Ti(nn.Sequential):
    def __init__(self,
                 inplanes: int = 3,
                 emb_size: int = 192,
                 num_heads: int = 3,
                 img_size: int = 224,
                 depth: int = 12,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(inplanes, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )


class ViT_S(nn.Sequential):
    def __init__(self,
                 inplanes: int = 3,
                 emb_size: int = 384,
                 num_heads: int = 6,
                 img_size: int = 224,
                 depth: int = 12,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(inplanes, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )


class ViT_B(nn.Sequential):
    def __init__(self,
                 inplanes: int = 3,
                 emb_size: int = 768,
                 num_heads: int = 12,
                 img_size: int = 224,
                 depth: int = 12,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(inplanes, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )


class ViT_L(nn.Sequential):
    def __init__(self,
                 inplanes: int = 3,
                 emb_size: int = 1024,
                 num_heads: int = 16,
                 img_size: int = 224,
                 depth: int = 24,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 **kwargs):
        super().__init__(
            PatchEmbedding(inplanes, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )

