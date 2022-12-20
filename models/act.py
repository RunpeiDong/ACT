# --------------------------------------------------------
# ACT Pretraining 3D Transformer Script
# Autoencoders as Cross-Modal Teachers
# Copyright (c) 2022 Runpei Dong
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightly import loss as losses
from timm.models.layers import DropPath, trunc_normal_
# from timm.loss import LabelSmoothingCrossEntropy

from .dvae import Group
from .dvae import DiscreteVAE, Encoder, ACTPromptedDiscreteVAEwithVIT
from .dvae import Encoder
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


class VisableOnlyMaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.embed_dim = config.transformer_config.embed_dim
        self.cls_dim = config.transformer_config.cls_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.dvae_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        if self.encoder_dims != self.embed_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.embed_dim)
        else:
            self.reduce_dim = nn.Identity()

        self.mask_type = config.transformer_config.mask_type
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.embed_dim, self.num_tokens)
        # head for cls contrast
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, register_shallow_hook=-1, only_cls_tokens=False, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C
        group_input_tokens = self.reduce_dim(group_input_tokens)

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        x_vis = torch.cat((cls_tokens, x_vis), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # transformer
        if register_shallow_hook > 0:
            for idx, blk in enumerate(self.blocks.blocks):
                x_vis = blk(x_vis + pos)
                if idx == register_shallow_hook:
                    x_vis_shallow = x_vis
        else:
            x_vis = self.blocks(x_vis, pos)
            x_vis_shallow = None
        x_vis = self.norm(x_vis)

        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            return self.cls_head(x_vis[:, 0])
        if register_shallow_hook > 0:
            return x_vis[:, 1:], x_vis[:, 0], x_vis_shallow[:, 1:], bool_masked_pos

        return x_vis[:, 1:], bool_masked_pos


class TokenAllMaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.mask_type = config.transformer_config.mask_type
        self.embed_dim = config.transformer_config.embed_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[Transformer args] {config.transformer_config}', logger='dVAE BERT')
        # define the encoder
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        # self.reduce_dim = nn.Linear(self.encoder_dims, self.embed_dim)

        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        # layer norm
        self.norm = nn.LayerNorm(self.embed_dim)
        # head for token classification
        # self.num_tokens = config.dvae_config.num_tokens
        # self.lm_head = nn.Linear(self.embed_dim, self.num_tokens)
        # head for cls contrast
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger='dVAE BERT')

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug=False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G  
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos

        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  # True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(
            detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1),
            detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0),
                                                                              detached_group_input_tokens.size(1),
                                                                              detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (
                    1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, neighborhood, center, register_shallow_hook=-1, return_all_tokens=False, only_cls_tokens=False,
                noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)  # B G
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        # group_input_tokens = self.reduce_dim(group_input_tokens)
        # replace the point
        replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone(),
                                                                         noaug=noaug)
        batch_size, seq_len, _ = replaced_group_input_tokens.size()
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        maksed_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, maksed_group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        if register_shallow_hook > 0:
            for idx, blk in enumerate(self.blocks.blocks):
                x = blk(x + pos)
                if idx == register_shallow_hook:
                    x_shallow = x
        else:
            x = self.blocks(x, pos)
            x_shallow = None
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            return self.cls_head(x[:, 0])
        if register_shallow_hook > 0:
            return x[:, 1:], x[:, 0], x_shallow[:, 1:], bool_masked_pos

        return x[:, 1:], bool_masked_pos
        # only return the cls feature, for moco contrast
        # if only_cls_tokens:
        #     return self.cls_head(x[:, 0])
        # logits = self.lm_head(x[:, 1:])
        # if return_all_tokens:
        #     return self.cls_head(x[:, 0]), logits
        # else:
        #     # return the flake logits
        #     return self.cls_head(x[:, 0]), logits[~overall_mask], logits[overall_mask], overall_mask # reduce the Batch dim


class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.mask_type = config.transformer_config.mask_type
        self.embed_dim = config.transformer_config.embed_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[Transformer args] {config.transformer_config}', logger='dVAE BERT')
        # define the encoder
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.embed_dim)

        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        # layer norm
        self.norm = nn.LayerNorm(self.embed_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.embed_dim, self.num_tokens)
        # head for cls contrast
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger='dVAE BERT')

    def _mask_center(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()

        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos

    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug=False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G  
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos

        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  # True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(
            detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1),
            detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0),
                                                                              detached_group_input_tokens.size(1),
                                                                              detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (
                    1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, neighborhood, center, return_all_tokens=False, only_cls_tokens=False, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center(center, noaug=noaug)  # B G
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # replace the point
        replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone(),
                                                                         noaug=noaug)
        batch_size, seq_len, _ = replaced_group_input_tokens.size()
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        maksed_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, maksed_group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            return self.cls_head(x[:, 0])
        logits = self.lm_head(x[:, 1:])
        if return_all_tokens:
            return self.cls_head(x[:, 0]), logits
        else:
            # return the flake logits
            return self.cls_head(x[:, 0]), logits[~overall_mask], logits[
                overall_mask], overall_mask  # reduce the Batch dim


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        if self.encoder_dims != self.embed_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.embed_dim)
        else:
            self.reduce_dim = nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.embed_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        if config.transfer_type == 'linear':
            # linear head
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.cls_dim)
            )
        else:
            # mlp-3 head (original head)
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.embed_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        # Side-Tuning
        self.setup_side()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        if config.transfer_type != 'full':
            # freeze all parameters except bias parameters (BitFit)
            for name, param in self.named_parameters():
                if config.transfer_type == 'mlp-3' or config.transfer_type == 'linear':
                    if not 'cls' in name:  # MLP-3/Linear
                        param.requires_grad = False  # not update by gradient
                elif config.transfer_type == 'side':
                    if not 'side' in name and not 'cls' in name:  # Side-Tuning
                        param.requires_grad = False  # not update by gradient
                elif config.transfer_type == 'bit-fit':
                    if not 'bias' in name and not 'cls' in name:  # BitFit
                        param.requires_grad = False  # not update by gradient

    def setup_side(self):
        if self.config.transfer_type != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.Tensor([0.0]))
            self.side = Encoder(encoder_channel=self.embed_dim)
            self.side_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        # self.loss_ce = LabelSmoothingCrossEntropy(smoothing=0.3)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path, custom_loading=False):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            if custom_loading:
                base_ckpt = {k.replace("module.point_encoder.", ""): v for k, v in ckpt['state_dict'].items()}
                # base_ckpt = {k.replace("module.decoder.cls.head", "cls_head_finetune"): v for k, v in base_ckpt.items()}
                # base_ckpt = {k.replace("cls_head_finetune.8", "cls_head_finetune.skip"): v for k, v in base_ckpt.items()}
                base_ckpt = {k.replace("encoder", "blocks"): v for k, v in base_ckpt.items()}
                base_ckpt = {k.replace("patch_embed", "encoder"): v for k, v in base_ckpt.items()}
            else:
                base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        if self.side is not None:
            side_output = self.side(neighborhood)
            side_output = self.side_projection(side_output)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            side_output = alpha_squashed * x[:, 1:] + (1 - alpha_squashed) * side_output
            concat_f = torch.cat([x[:, 0], side_output.max(1)[0]], dim=-1)
        else:
            concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)

        ret = self.cls_head_finetune(concat_f)
        return ret


@MODELS.register_module()
class ACT_PointBERT(nn.Module):
    """ PointBERT with ACT
    """

    def __init__(self, config):
        super().__init__()
        print_log(f'[ACT] build dVAE_BERT ...', logger='ACT')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K

        self.moco_loss = config.transformer_config.moco_loss
        self.dvae_loss = config.transformer_config.dvae_loss
        self.cutmix_loss = config.transformer_config.cutmix_loss

        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[ACT] PointTranslator calc the loss for all token ...', logger='ACT')
        else:
            print_log(f'[ACT] PointTranslator [NOT] calc the loss for all token ...', logger='ACT')

        self.transformer_q = MaskTransformer(config)
        self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)

        self.transformer_k = MaskTransformer(config)
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.dvae = ACTPromptedDiscreteVAEwithVIT(config.dvae_config)
        self._prepare_dvae()

        for param in self.dvae.parameters():
            param.requires_grad = False

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        print_log(
            f'[Point_BERT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...',
            logger='ACT')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # create the queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # loss
        self.build_loss_func()

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger='ACT')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens=True, noaug=True)
            return cls_feature

    def _mixup_pc(self, neighborhood, center, dvae_label):
        '''
            neighborhood : B G M 3
            center: B G 3
            dvae_label: B G
            ----------------------
            mixup_ratio: /alpha:
                mixup_label = alpha * origin + (1 - alpha) * flip

        '''
        mixup_ratio = torch.rand(neighborhood.size(0))
        mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.type_as(neighborhood)
        mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (
                    1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))
        mixup_dvae_label = dvae_label * mixup_mask + dvae_label.flip(0) * (1 - mixup_mask)

        return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center, mixup_dvae_label.long()

    def forward(self, pts, noaug=False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        else:
            # divide the point cloud in the same form. This is important
            neighborhood, center = self.group_divider(pts)
            # produce the gt point tokens
            with torch.no_grad():
                dvae_label = self.dvae.forward_tokenizer(neighborhood, center)
            # forward the query model in mask style 1.
            if self.return_all_tokens:
                q_cls_feature, logits = self.transformer_q(neighborhood, center,
                                                           return_all_tokens=self.return_all_tokens)  # logits :  N G C
            else:
                q_cls_feature, real_logits, flake_logits, mask = self.transformer_q(neighborhood, center,
                                                                                    return_all_tokens=self.return_all_tokens)  # logits :  N' C where N' is the mask.sum()
            q_cls_feature = nn.functional.normalize(q_cls_feature, dim=1)

            mixup_ratio, mixup_neighborhood, mixup_center, mix_dvae_label = self._mixup_pc(neighborhood, center,
                                                                                           dvae_label)
            if self.return_all_tokens:
                mixup_cls_feature, mixup_logits = self.transformer_q(mixup_neighborhood, mixup_center,
                                                                     return_all_tokens=self.return_all_tokens)
            else:
                mixup_cls_feature, mixup_real_logits, mixup_flake_logits, mixup_mask = self.transformer_q(
                    mixup_neighborhood, mixup_center, return_all_tokens=self.return_all_tokens)
            mixup_cls_feature = nn.functional.normalize(mixup_cls_feature, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k_cls_feature = self.transformer_k(neighborhood, center, only_cls_tokens=True)  # keys: NxC
                k_cls_feature = nn.functional.normalize(k_cls_feature, dim=1)

            if self.moco_loss:
                # ce loss with moco contrast
                l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1)  # n 1
                l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()])  # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
                moco_loss = self.loss_ce(ce_logits, labels)
            else:
                moco_loss = torch.tensor(0.).to(pts.device)

            if self.dvae_loss:
                if self.return_all_tokens:
                    dvae_loss = self.loss_ce(logits.reshape(-1, logits.size(-1)), dvae_label.reshape(-1, )) + \
                                self.loss_ce(mixup_logits.reshape(-1, mixup_logits.size(-1)),
                                             mix_dvae_label.reshape(-1, ))
                else:
                    dvae_loss = self.loss_ce(flake_logits, dvae_label[mask]) + \
                                self.loss_ce(mixup_flake_logits, mix_dvae_label[mixup_mask])
            else:
                dvae_loss = torch.tensor(0.).to(pts.device)

            if self.cutmix_loss:
                l_pos = torch.einsum('nc, mc->nm', [mixup_cls_feature, k_cls_feature])  # n n
                l_neg = torch.einsum('nc, ck->nk', [mixup_cls_feature, self.queue.clone().detach()])  # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.arange(l_pos.shape[0], dtype=torch.long).to(pts.device)
                cutmix_loss = (mixup_ratio * self.loss_ce_batch(ce_logits, labels) + (
                            1 - mixup_ratio) * self.loss_ce_batch(ce_logits, labels.flip(0))).mean()
            else:
                cutmix_loss = torch.tensor(0.).to(pts.device)
            self._dequeue_and_enqueue(k_cls_feature)
            # return moco_loss + dvae_loss, cutmix_loss
            return moco_loss, dvae_loss, cutmix_loss
            # return moco_loss + dvae_loss + cutmix_loss


@MODELS.register_module()
class ACT_PointDistillation(nn.Module):
    """ ACT for Point Feature Distillation
    """

    def __init__(self, config):
        super().__init__()
        print_log(f'[ACT] build Transformer for feature distillation pretraining', logger='ACT')
        self.config = config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.embed_dim = config.transformer_config.embed_dim
        self.ACT_encoder = VisableOnlyMaskTransformer(config)
        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group
        self.proj_type = config.transformer_config.proj
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        self.cls_loss = config.transformer_config.cls_loss
        self.register_shallow_hook = config.transformer_config.register_shallow_hook

        if self.cls_loss:
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            trunc_normal_(self.cls_pos, std=.02)

        # tokenizer
        self.build_tokenizer(config.dvae_config)

        # grouper
        print_log(f'[ACT] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='ACT')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        if self.proj_type == 'conv':
            self.proj_head = nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, 1),
                # nn.BatchNorm1d(1024),
                # nn.LeakyReLU(negative_slope=0.2),
                # nn.Conv1d(self.embed_dim, 3*self.group_size, 1)
            )
        elif self.proj_type == 'linear':
            self.proj_head = nn.Linear(self.embed_dim, config.dvae_config.tokens_dims)
        else:
            self.proj_head = nn.Identity()

        # decoder
        self.build_masked_decoder()
        # loss
        self.loss_type = config.loss
        self.build_loss_func()

    def build_tokenizer(self, cfg):
        self.dvae_tokenizer = ACTPromptedDiscreteVAEwithVIT(cfg)
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae_tokenizer.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger='ACT')

        for param in self.dvae_tokenizer.parameters():
            param.requires_grad = False

    def build_masked_decoder(self):
        if self.mask_ratio > 0.:
            print_log(f'[ACT] build masked decoder for feature prediction ...', logger='ACT')
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.decoder_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.embed_dim)
            )
            dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
            self.ACT_decoder = TransformerDecoder(
                embed_dim=self.embed_dim,
                depth=self.decoder_depth,
                drop_path_rate=dpr,
                num_heads=self.decoder_num_heads,
            )
            trunc_normal_(self.mask_token, std=.02)
        else:
            print_log(f'[ACT] pretraining without masked decoder ...', logger='ACT')
            self.mask_token = None
            self.ACT_decoder = None

    def build_loss_func(self):
        print_log(f"[ACT] build {self.loss_type} loss function for pretraining ...", logger='ACT')
        if self.loss_type == 'cosine':
            self.loss_func = losses.NegativeCosineSimilarity()
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif self.loss_type == 'smoothl1':
            self.loss_func = nn.SmoothL1Loss(reduction='mean')
        elif self.loss_type == 'ntxent':
            self.loss_func = losses.NTXentLoss(temperature=0.07)
        elif self.loss_type == 'barlow':
            self.loss_func = losses.BarlowTwinsLoss(lambda_param=5.e-3)

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature = self.ACT_encoder(neighborhood, center, only_cls_tokens=True, noaug=True)
            return cls_feature

    def forward(self, pts, noaug=False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        neighborhood, center = self.group_divider(pts)

        if self.cls_loss:
            x_vis, x_vis_cls, x_vis_shallow, mask = self.ACT_encoder(neighborhood, center,
                                                                     register_shallow_hook=self.register_shallow_hook)
        else:
            x_vis, mask = self.ACT_encoder(neighborhood, center)
            x_vis_cls, x_vis_shallow = None, None
        batch_size, seq_length, channel = x_vis.shape

        with torch.no_grad():
            teacher_feat = self.dvae_tokenizer.forward_tokenizer_features(neighborhood, center, return_global=True)

        if self.mask_token is not None:
            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(batch_size, -1, channel)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(batch_size, -1, channel)
            _, num_mask, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(batch_size, num_mask, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            x_vis = self.ACT_decoder(x_full, pos_full, num_mask)
            student_feat = self.proj_head(x_vis)
            teacher_feat = teacher_feat[mask].reshape(batch_size, -1, student_feat.shape[-1])
            if self.cls_loss:
                x_full_shallow = torch.cat([x_vis_cls.unsqueeze(1), x_vis_shallow, mask_token], dim=1)
                cls_pos = self.cls_pos.expand(batch_size, -1, -1)
                pos_full_shallow = torch.cat([cls_pos, pos_full], dim=1)
                x_vis_shallow = self.ACT_decoder(x_full_shallow, pos_full_shallow, num_mask)
                student_feat_global = self.proj_head(x_vis_shallow)
            else:
                student_feat_global = None
        else:
            num_mask = 1
            student_feat = self.proj_head(x_vis)

        assert teacher_feat.shape == student_feat.shape
        loss = torch.FloatTensor([0.0]).cuda()
        if not 'l1' in self.loss_type and not 'l2' in self.loss_type:
            for batch_id in range(batch_size):
                if self.loss_type == 'cosine':
                    loss += (1 + self.loss_func(student_feat[batch_id], teacher_feat[batch_id]).mean())
                    if student_feat_global is not None:
                        loss += (1 + self.loss_func(student_feat_global[batch_id], teacher_feat[batch_id]).mean())
                else:
                    loss += (self.loss_func(student_feat[batch_id], teacher_feat[batch_id]) / num_mask)
                    if student_feat_global is not None:
                        loss += (self.loss_func(student_feat_global[batch_id], teacher_feat[batch_id]) / num_mask)
            loss = loss.mean() / batch_size
        else:
            loss = self.loss_func(student_feat, teacher_feat)

        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
