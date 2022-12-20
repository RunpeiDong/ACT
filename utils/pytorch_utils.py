r''' Tiny PointNetV2 Patch Embedding Module Utility
Copyright (c) 2022, Runpei Dong. All rights reserved.
Licensed under The MIT License [see LICENSE for details]

Implementation based on Pointnet2_PyTorch and PointMLP code bases:
https://github.com/erikwijmans/Pointnet2_PyTorch
https://github.com/ma-xu/pointMLP-pytorch
'''
import torch
import torch.nn as nn

import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from typing import List, Tuple
from .transformer_layers import DropPath

### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm:
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNetFeaturePropagationRes(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagationRes, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        extraction = []
        for _ in range(blocks):
            extraction.append(
                ConvBNReLURes1D(out_channel, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.extraction = nn.Sequential(*extraction)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, C', N]
            points2: input points data, [B, C'', S]
        Return:
            new_points: upsampled points data, [B, C''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points
    

class LinearBottleNeck(nn.Module):
    
    def __init__(self, channel, middle_channel, drop_path=0., gamma=None, act='gelu'):
        super().__init__()
        self.linear_down = nn.Linear(channel, middle_channel)
        self.norm = nn.LayerNorm(channel, eps=1e-6)
        self.linear_up = nn.Linear(middle_channel, channel)
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()


class ConvNeXtBottleNeck(nn.Module):
    
    def __init__(self, channel, middle_channel, out_channel=384, drop_path=0., gamma=None, act='gelu'):
        super().__init__()
        self.dwconv = nn.Conv1d(channel, channel, kernel_size=7, padding=3, groups=channel)
        self.norm = nn.LayerNorm(channel, eps=1e-6)
        self.pwconv1 = nn.Linear(channel, middle_channel)  # pointwise/1x1 convs, implemented with linear layers
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(middle_channel, out_channel)
        self.gamma = gamma
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 1).contiguous() # (B, L, C) -> (B, C, L)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1).contiguous() # (B, C, L) -> (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        if x.shape != input.shape:
            return self.drop_path(x)
        x = input + self.drop_path(x)
        return x

class ConvBNReLURes1DBottleNeck(nn.Module):
    def __init__(self, channel, middle_channel=None, out_channel=None, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLURes1DBottleNeck, self).__init__()
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.Identity()
        self.middle_channel = middle_channel if middle_channel else int(channel * res_expansion)
        self.out_channel = out_channel if out_channel else channel
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=self.middle_channel,
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(self.middle_channel),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=self.middle_channel, out_channels=self.out_channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(self.out_channel),
                self.act,
                nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(self.out_channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=self.middle_channel, out_channels=self.out_channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(self.out_channel)
            )

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

    def forward(self, x, transpose=True):
        if x.ndim != 3:
            x = x.unsqueeze(0)
        if transpose:
            x = x.permute(0, 2, 1).contiguous()
            return self.act(self.net2(self.net1(x)) + x).permute(0, 2, 1).contiguous()
        return self.act(self.net2(self.net1(x)) + x)
    
class ConvBNReLU1DBottleNeck(nn.Module):
    def __init__(self, channel, middle_channel=None, out_channel=None, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLU1DBottleNeck, self).__init__()
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.Identity()
        self.middle_channel = middle_channel if middle_channel else int(channel * res_expansion)
        self.out_channel = out_channel if out_channel else channel
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=self.middle_channel,
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(self.middle_channel),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=self.middle_channel, out_channels=self.out_channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(self.out_channel),
                self.act,
                nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(self.out_channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=self.middle_channel, out_channels=self.out_channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(self.out_channel)
            )

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

    def forward(self, x, transpose=True):
        if x.ndim != 3:
            x = x.unsqueeze(0)
        if transpose:
            x = x.permute(0, 2, 1).contiguous()
            return self.act(self.net2(self.net1(x))).permute(0, 2, 1).contiguous()
        return self.act(self.net2(self.net1(x)))

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x, transpose=True):
        if transpose:
            x = x.permute(0, 2, 1).contiguous()
            return self.net(x).permute(0, 2, 1).contiguous()
        return self.net(x)

class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )
            
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

    def forward(self, x, transpose=True):
        if x.ndim != 3:
            x = x.unsqueeze(0)
        if transpose:
            x = x.permute(0, 2, 1).contiguous()
            return self.act(self.net2(self.net1(x)) + x).permute(0, 2, 1).contiguous()
        return self.act(self.net2(self.net1(x)) + x)

class ConvBNReLURes(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True):
        super(ConvBNReLURes, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm2d(channel),
                self.act,
                nn.Conv2d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(channel)
            )
            
        self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class SharedPointMLP(nn.Module):

    def __init__(
        self, 
        args: List[int],
        res_blocks: List[int] = [2, 2]
    ):
        super().__init__()

        self.patch_embed_dim = args[-1]

        self.first_conv = nn.Sequential(
            ConvBNReLU1D(args[0], args[1], 1),
            ConvBNReLU1D(args[1], args[2], 1)
        )

        self.res_conv_pre = []
        for _ in range(res_blocks[0]):
            self.res_conv_pre.append(ConvBNReLURes1D(channel=args[2]))
        self.res_conv_pre = nn.Sequential(*self.res_conv_pre)

        self.res_conv_post = []
        for _ in range(res_blocks[1]):
            self.res_conv_post.append(ConvBNReLURes1D(channel=args[2]))
        self.res_conv_post = nn.Sequential(*self.res_conv_post)

        self.second_conv = nn.Sequential(
            ConvBNReLU1D(args[2], self.patch_embed_dim, 1)
        )

    def forward(self, grouped_feat):
        # input: (B, C, npoint, nsample) (B x 3 x 1024 x 64)
        # output: (B, mlp[-1], npoint, nsample) 
        B, C, npoint, nsample = grouped_feat.shape
        # B C G S -> B G C S -> BG C S
        grouped_feat = grouped_feat.permute(0, 2, 1, 3).contiguous().reshape(B * npoint, C, nsample) # BG C 64
        feature = self.first_conv(grouped_feat)  # BG 256 64
        if len(self.res_conv_pre) > 0:
            feature = self.res_conv_pre(feature)
        feature = F.adaptive_max_pool1d(feature, 1).view(B * npoint, -1)
        # BG C -> B G C -> B C G
        feature = feature.reshape(B, npoint, -1).permute(0, 2, 1)
        if len(self.res_conv_post) > 0:
            feature = self.res_conv_post(feature)
        feature = self.second_conv(feature)
        
        return feature.unsqueeze(-1)

class SharedMLPResidual(nn.Module):
    
    def __init__(
        self, 
        args: List[int],
        res_blocks: List[int] = [2, 2]
    ):
        super().__init__()

        self.patch_embed_dim = args[-1]

        self.first_conv = nn.Sequential(
            nn.Conv1d(args[0], args[1], 1),
            nn.BatchNorm1d(args[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(args[1], args[2], 1)
        )
        self.res_conv_pre = []
        for _ in range(res_blocks[0]):
            self.res_conv_pre.append(ConvBNReLURes1D(channel=args[2]))
        self.res_conv_pre = nn.Sequential(*self.res_conv_pre)
        self.second_conv = nn.Sequential(
            nn.Conv1d(args[2]*2, args[2]*2, 1),
            nn.BatchNorm1d(args[2]*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(args[2]*2, self.patch_embed_dim, 1)
        )
        self.res_conv_post = []
        for _ in range(res_blocks[1]):
            self.res_conv_post.append(ConvBNReLURes1D(channel=self.patch_embed_dim*2))
        self.res_conv_post = nn.Sequential(*self.res_conv_post)

    def forward(self, grouped_feat):
        B, C, npoint, nsample = grouped_feat.shape
        grouped_feat = grouped_feat.permute(0, 2, 1, 3).contiguous().reshape(B * npoint, C, nsample) # BG C 64
        feature = self.first_conv(grouped_feat)  # BG 256 64
        if len(self.res_conv_pre) > 0:
            feature = self.res_conv_pre(feature)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, nsample), feature], dim=1)# BG 512 n
        if len(self.res_conv_post) > 0:
            feature = self.res_conv_post(feature)
        feature = self.second_conv(feature) # BG 1024 n
        feature = feature.reshape(B, npoint, self.patch_embed_dim, nsample).permute(0, 2, 1, 3)
        return feature

class SharedMLPWithPooling(nn.Module):

    def __init__(
        self, 
        args: List[int]
    ):
        super().__init__()

        self.patch_embed_dim = args[-1]

        # [3, 128, 256, 512, patch_embed_dim]
        self.first_conv = nn.Sequential(
            nn.Conv1d(args[0], args[1], 1),
            nn.BatchNorm1d(args[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(args[1], args[2], 1)
        )
        # self.second_res_conv = ConvBNReLU1D(channel=args[2])
        self.second_conv = nn.Sequential(
            nn.Conv1d(args[2]*2, args[2]*2, 1),
            nn.BatchNorm1d(args[2]*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(args[2]*2, self.patch_embed_dim, 1)
        )

    def forward(self, grouped_feat):
        # input: (B, C, npoint, nsample) (B x 3 x 1024 x 64)
        # output: (B, mlp[-1], npoint, nsample) 
        B, C, npoint, nsample = grouped_feat.shape
        grouped_feat = grouped_feat.permute(0, 2, 1, 3).contiguous().reshape(B * npoint, C, nsample) # BG C 64
        feature = self.first_conv(grouped_feat)  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, nsample), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature = feature.reshape(B, npoint, self.patch_embed_dim, nsample).permute(0, 2, 1, 3)
        # feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        # feature_global = feature_global.reshape(B, G, self.patch_embed_dim)
        return feature


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )
    
    def forward(self, input):
        _type_checked = False
        for module in self:
            # XXX type checking to support FP16 training, it may not be the best practice
            if not _type_checked and (isinstance(module.conv.weight, torch.HalfTensor) or 
                                      isinstance(module.conv.weight, torch.cuda.HalfTensor)):
                input = input.half().cuda()
                _type_checked = True
            input = module(input)
        return input


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv3d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int, int] = (1, 1, 1),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (0, 0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


