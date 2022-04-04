"""
Restart: Efficient Architecture on Time-Delay Neural Network.

Author: Rui Wang
Version: v0.1
Date: 2021.2.2 20:27

Version: v0.2
Date: 2021.2.9 16:02

PS: Hard to start, but importantly to the first step.
"""

import copy
import os
import yaml
from collections import OrderedDict
from typing import Union

import numpy as np
import geatpy as ea
import torch
import torch.nn as nn
import torch.nn.functional as F
from sugar.metrics import profile
from sugar.modules.dynamicarch import DynamicBatchNorm1d, DynamicConv1d, DynamicLinear, DynamicModule


__all__ = ['tdnn6m2g', 'tdnn14m4g', 'tdnn26m7g', 'ArchCoder', 'AccuracyPredictor']


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _prod(sizes):
    """
    Product of tiny list of sizes. It is faster than numpy.prod and torch.prod.
        
        Parameter
        ---------
            sizes : list or tuple
                Size of inputs, output, or weights, usually 2/3/4 dimensions.

        Performance
        -----------
            profile : 20 it/s
            torch.prod : 500 it/s
            numpy.prod : 700 it/s
            _prod : 1500 it/s
    """
    ans = 1
    for s in sizes:
        ans *= s
    return ans

@torch.no_grad()
def macs_params(model, input_size=(1, 80, 300), custom_ops=None, verbose=True, device='cpu'):
    """Count MACs (multiply-and-accumulates) and Params (parameters) using thop."""
    macs, params = profile(model, input_size=input_size, custom_ops=custom_ops, verbose=verbose, device=device)
    if verbose:
        print('{}: MACs {} and Params {} using input of {}'.format(model.__class__.__name__, macs, params, input_size))
    return macs, params


class DynamicSELayer(DynamicModule):
    def __init__(self, in_channels, reduction=4) -> None:
        super(DynamicSELayer, self).__init__()
        assert in_channels % reduction == 0, "{} % {} != 0".format(in_channels, reduction)

        self.in_channels = in_channels
        self.reduction = reduction
        self.reduce = DynamicLinear(in_channels, in_channels // reduction)
        self.expand = DynamicLinear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

        self.active_in_channels = in_channels

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.reduce(out))
        out = self.sigmoid(self.expand(out))
        B, C, _ = x.size()
        out = x * out.view(B, C, 1)
        return out

    @property
    def _module_configs(self):
        """Config is devired from active config"""
        config_reduce = (self.active_in_channels, self.active_in_channels // self.reduction)
        config_expand = (self.active_in_channels // self.reduction, self.active_in_channels)
        return config_reduce, config_expand

    @property
    def config(self):
        """
        [DynamicSELayer] Config: in_channels
        """
        return self.active_in_channels

    @config.setter
    def config(self, in_channels):
        self.active_in_channels = self.in_channels if in_channels is None else in_channels
        config_reduce, config_expand = self._module_configs
        self.reduce.config = config_reduce
        self.expand.config = config_expand

    def clone(self, config=None):
        self.config = config
        config_reduce, config_expand = self._module_configs

        m = DynamicSELayer(self.active_in_channels, self.reduction)
        m = m.to(self.device)

        m.add_module('reduce', self.reduce.clone(config_reduce))
        m.add_module('expand', self.expand.clone(config_expand))
        
        return m.train(self.training)

    def out_size(self, in_size):
        return in_size

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)

            Notes
            -----
                MACs consider the forwards of reduce and expand, but the operations of mean,
                sigmoid, and * are ignores. 
        """
        reduce_in_size = in_size[:2]
        reduce_ops = self.reduce.count_ops(reduce_in_size)
        expand_in_size = self.reduce.out_size(reduce_in_size)
        expand_ops = self.expand.count_ops(expand_in_size)

        total_ops = reduce_ops + expand_ops
        return total_ops

    def count_params(self):
        reduce_params = self.reduce.count_params()
        expand_params = self.expand.count_params()
        total_params = reduce_params + expand_params
        return total_params

    @property
    def device(self):
        return self.reduce.device


class DynamicConv1dReLUBN(DynamicModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicConv1dReLUBN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.conv = DynamicConv1d(in_channels, out_channels, kernel_size, stride, dilation)
        self.bn = DynamicBatchNorm1d(out_channels)

        self.active_kernel_size = kernel_size
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))
    
    @property
    def _module_configs(self):
        config_conv = (self.active_in_channels, self.active_out_channels, self.active_kernel_size)
        config_bn = self.active_out_channels
        return config_conv, config_bn

    @property
    def config(self):
        """
        [DynamicConv1dReLUBN] Config: (in_channels, out_channels, kernel_size)
            
            Notes
            -----
                The in_channels or out_channels can be **indices** when they are as before or after Res2TDNN layer.
        """
        return (self.active_in_channels, self.active_out_channels, self.active_kernel_size)

    @config.setter
    def config(self, config):
        if config is None:
            config = (None, None, None)
        in_channels, out_channels, kernel_size = config

        self.active_in_channels = self.in_channels if in_channels is None else in_channels
        self.active_out_channels = self.out_channels if out_channels is None else out_channels
        self.active_kernel_size = self.kernel_size if kernel_size is None else kernel_size

        config_conv, config_bn = self._module_configs
        self.conv.config = config_conv
        self.bn.config = config_bn

    def clone(self, config=None):
        self.config = config

        in_channels = self.active_in_channels if isinstance(self.active_in_channels, int) else self.active_in_channels.numel()
        out_channels = self.active_out_channels if isinstance(self.active_out_channels, int) else self.active_out_channels.numel()

        m = DynamicConv1dReLUBN(in_channels, out_channels, self.active_kernel_size, self.stride, self.dilation)
        m = m.to(self.device)

        config_conv, config_bn = self._module_configs
        m.add_module('conv', self.conv.clone(config_conv))
        m.add_module('bn', self.bn.clone(config_bn))

        return m.train(self.training)

    def out_size(self, in_size):
        return self.conv.out_size(in_size)

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)

            Notes
            -----
                MACs are from conv1d and bn, but relu is 0.
        """
        conv_ops = self.conv.count_ops(in_size)
        bn_in_size = self.conv.out_size(in_size)
        bn_ops = self.bn.count_ops(bn_in_size)

        total_ops = conv_ops + bn_ops
        return total_ops

    def count_params(self):
        conv_params = self.conv.count_params()
        bn_params = self.bn.count_params()
        total_params = conv_params + bn_params
        return total_params

    @property
    def device(self):
        return self.conv.device


class DynamicRes2Conv1dReLUBN(DynamicModule):
    def __init__(self, channels, scale=8, kernel_size=1, stride=1, dilation=1):
        """
            Notes
            -----
                scale == 1 results in bottleneck connection.
        """
        super(DynamicRes2Conv1dReLUBN, self).__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)

        self.scale = scale
        self.nums = scale if scale == 1 else scale - 1
        self.width = channels // scale
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.convs = []
        for _ in range(self.nums):
            m = nn.Sequential(OrderedDict([
                ('conv', DynamicConv1d(self.width, self.width, kernel_size, stride, dilation)), 
                ('relu', nn.ReLU()), 
                ('bn', DynamicBatchNorm1d(self.width))
            ]))
            self.convs.append(m)
        self.convs = nn.ModuleList(self.convs)

        self.active_width = self.width
        self.active_kernel_size = self.kernel_size
    
    def forward(self, x):
        spx = torch.split(x, self.active_width, 1)
        out = []
        sp = spx[0]
        for i in range(self.nums):
            if i == 0:
                sp = self.convs[i](spx[i])
            else:
                sp = self.convs[i](spx[i] + sp)
            out.append(sp)
        if self.scale > 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out

    @property
    def _module_configs(self):
        config_convs_conv_i = (self.active_width, self.active_width, self.active_kernel_size)
        config_convs_bn_i = self.active_width
        return config_convs_conv_i, config_convs_bn_i

    @property
    def config(self):
        """
        [DynamicRes2Conv1dReLUBN] Config: (width, kernel_size)
        """
        return (self.active_width, self.active_kernel_size)

    @config.setter
    def config(self, config):
        if config is None:
            config = (None, None)
        width, kernel_size = config

        self.active_width = self.width if width is None else width
        self.active_kernel_size = self.kernel_size if kernel_size is None else kernel_size
        
        config_convs_conv_i, config_convs_bn_i = self._module_configs
        for i in range(self.nums):
            self.convs[i].conv.config = config_convs_conv_i
            self.convs[i].bn.config = config_convs_bn_i

    def clone(self, config=None):
        self.config = config

        m = DynamicRes2Conv1dReLUBN(self.active_width * self.scale, self.scale, self.active_kernel_size, self.stride, self.dilation)
        m = m.to(self.device)

        config_convs_conv_i, config_convs_bn_i = self._module_configs
        convs = []
        for i in range(self.nums):
            conv = self.convs[i].conv.clone(config_convs_conv_i)
            bn = self.convs[i].bn.clone(config_convs_bn_i)
            convs.append(nn.Sequential(OrderedDict([
                ('conv', conv), 
                ('relu', nn.ReLU()), 
                ('bn', bn)
            ])))
        m.add_module('convs', nn.ModuleList(convs))

        return m.train(self.training)

    def out_size(self, in_size):
        return in_size

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)

            Notes
            -----
                MACs are from conv1d and bn, but relu is 0.
        """
        res2_in_size = in_size[:1] + [self.active_width] + in_size[2:]
        res2_ops = 0
        for i in range(self.nums):
            res2_ops = res2_ops + self.convs[i].conv.count_ops(res2_in_size)
            res2_ops = res2_ops + self.convs[i].bn.count_ops(res2_in_size)

        total_ops = res2_ops
        return total_ops

    def count_params(self):
        total_params = 0.0
        for i in range(self.nums):
            total_params = total_params + self.convs[i].conv.count_params() + self.convs[i].bn.count_params()
        return total_params

    @property
    def device(self):
        return self.convs[0].conv.device


class DynamicSERes2DilatedTDNNBlock(DynamicModule):
    def __init__(self, channels, mid_channels, kernel_size, dilation, scale=8):
        super(DynamicSERes2DilatedTDNNBlock, self).__init__()
        self.channels = channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = scale

        self.first_conv = DynamicConv1dReLUBN(channels, mid_channels)
        self.resconv = DynamicRes2Conv1dReLUBN(mid_channels, scale, kernel_size, dilation=dilation)
        self.last_conv = DynamicConv1dReLUBN(mid_channels, channels)
        self.se = DynamicSELayer(channels)

        self.active_channels = channels
        self.active_mid_channels = mid_channels
        self.active_kernel_size = kernel_size

    def forward(self, x):
        out = self.first_conv(x)
        out = self.resconv(out)
        out = self.last_conv(out)
        out = self.se(out)
        return out

    @property
    def _module_configs(self):
        active_width = self.active_mid_channels // self.resconv.scale
        mid_indices = torch.arange(self.mid_channels, device=self.device).view(self.resconv.scale, -1)[:, :active_width].flatten().long()

        config_first = (self.active_channels, mid_indices, None)
        config_resconv = (active_width, self.active_kernel_size)
        config_last = (mid_indices, self.active_channels, None)
        config_se = self.active_channels
        return config_first, config_resconv, config_last, config_se

    @property
    def config(self):
        """
        [DynamicSERes2DilatedTDNNBlock] Config: (channels, mid_channels, kernel_size)
        """
        return (self.active_channels, self.active_mid_channels, self.active_kernel_size)

    @config.setter
    def config(self, config):
        if config is None:
            config = (None, None, None)
        channels, mid_channels, kernel_size = config

        self.active_channels = self.channels if channels is None else channels
        self.active_mid_channels = self.mid_channels if mid_channels is None else mid_channels
        self.active_kernel_size = self.kernel_size if kernel_size is None else kernel_size

        config_first, config_resconv, config_last, config_se = self._module_configs
        
        self.first_conv.config = config_first
        self.resconv.config = config_resconv
        self.last_conv.config = config_last
        self.se.config = config_se

    def clone(self, config=None):
        self.config = config
        m = DynamicSERes2DilatedTDNNBlock(self.active_channels, self.active_mid_channels, self.active_kernel_size, self.dilation, self.scale)
        m = m.to(self.device)

        config_first, config_resconv, config_last, config_se = self._module_configs

        m.add_module('first_conv', self.first_conv.clone(config_first))
        m.add_module('resconv', self.resconv.clone(config_resconv))
        m.add_module('last_conv', self.last_conv.clone(config_last))
        m.add_module('se', self.se.clone(config_se))

        return m.train(self.training)

    def out_size(self, in_size):
        return in_size

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)

            Notes
            -----
                MACs are from first_conv, resconv, last_conv, and se.
        """
        resconv_in_size = self.first_conv.out_size(in_size)
        last_conv_in_size = self.resconv.out_size(resconv_in_size)
        se_in_size = self.last_conv.out_size(last_conv_in_size)

        first_conv_ops = self.first_conv.count_ops(in_size)
        resconv_ops = self.resconv.count_ops(resconv_in_size)
        last_conv_ops = self.last_conv.count_ops(last_conv_in_size)
        se_ops = self.se.count_ops(se_in_size)

        total_ops = first_conv_ops + resconv_ops + last_conv_ops + se_ops
        return total_ops

    def count_params(self):
        first_conv_params = self.first_conv.count_params()
        resconv_params = self.resconv.count_params()
        last_conv_params = self.last_conv.count_params()
        se_params = self.se.count_params()
        total_params = first_conv_params + resconv_params + last_conv_params + se_params
        return total_params

    @property
    def device(self):
        return self.resconv.device


class DynamicAttentiveStatsPool(DynamicModule):
    def __init__(self, in_dim, bottleneck_dim=128):
        super(DynamicAttentiveStatsPool, self).__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim
        self.linear1 = DynamicConv1d(in_dim, bottleneck_dim)
        self.linear2 = DynamicConv1d(bottleneck_dim, in_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(2)

        self.active_in_channels = self.in_dim

    def forward(self, x):
        """
            x : (batch, C, T)
        """
        alpha = self.tanh(self.linear1(x)) # Tanh: 4 exp + 2 add + 1 div
        alpha = self.softmax(self.linear2(alpha))
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        out = torch.cat([mean, std], dim=1)
        return out

    @property
    def _module_configs(self):
        config_linear1 = (self.active_in_channels, self.bottleneck_dim, None)
        config_linear2 = (self.bottleneck_dim, self.active_in_channels, None)
        return config_linear1, config_linear2
    
    @property
    def config(self):
        """
        [DynamicAttentiveStatsPool] Config: in_channels
        """
        return self.active_in_channels

    @config.setter
    def config(self, config):
        """
            config : int
        """
        in_channels = config

        self.active_in_channels = in_channels

        config_linear1, config_linear2 = self._module_configs

        self.linear1.config = config_linear1
        self.linear2.config = config_linear2

    def clone(self, config=None):
        self.config = config
        m = DynamicAttentiveStatsPool(self.active_in_channels, self.bottleneck_dim)
        m = m.to(self.device)

        config_linear1, config_linear2 = self._module_configs

        m.add_module('linear1', self.linear1.clone(config_linear1))
        m.add_module('linear2', self.linear2.clone(config_linear2))

        return m.train(self.training)

    def out_size(self, in_size):
        return in_size[:1] + [in_size[1] * 2]

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)

            Notes
            -----
                MACs are from linear1 and linear2, but Tanh, softmax, sum, ** 2, *, 
                and sqrt are ignored.
        """
        linear2_in_size = self.linear1.out_size(in_size)

        linear1_ops = self.linear1.count_ops(in_size)
        linear2_ops = self.linear2.count_ops(linear2_in_size)

        total_ops = linear1_ops + linear2_ops
        return total_ops

    def count_params(self):
        linear1_params = self.linear1.count_params()
        linear2_params = self.linear2.count_params()
        total_params = linear1_params + linear2_params
        return total_params

    @property
    def device(self):
        return self.linear1.device


class DynamicTDNN(DynamicModule):
    """
    A dynamic TDNN with residual connection, dense connection, and res2net block.
        The mainly parameters are in the first conv layer (1M+), a series of blocks(2M+), cat conv layer(2M+).
    """
    def __init__(self, 
                 in_feats: int = 80, 
                 out_embeds: int = 192, 
                 num_blocks: int = 3, 
                 channels: Union[int, tuple, list] = 512,
                 kernel_sizes: Union[int, tuple, list] = 5,
                 catconv_channels: int = 1536,
                 search_space=None,
                 kernels=[1, 3, 5],
                 depths=[-2, -1, 0],
                 widths=[0.25, 0.35, 0.5, 0.75, 1.0]):
        """
            Parameters
            ----------
                in_feats: dim of input's features
                out_embeds: dim of output's embeddings
                (dynamic) num_blocks: num of middle TDNN blocks
                (dynamic) channels: num of channels for layer1, 1st/middle layers of TDNN blocks, input of catconv, where the 1st value is for 1st layers of blocks, and the rest values are for layers of blocks.
                (dynamic) kernel_sizes: size of kernel for middle layers of TDNN blocks
                (dynamic) catconv_channels: num of catconv output's channels as well as pooling and linear
        """
        assert isinstance(channels, int) or len(channels) == num_blocks + 1
        assert isinstance(kernel_sizes, int) or len(kernel_sizes) == num_blocks + 1
                
        super(DynamicTDNN, self).__init__()
        self.in_feats = in_feats
        self.out_embeds = out_embeds
        self.num_blocks = num_blocks
        self.channels = [channels for _ in range(num_blocks + 1)] if isinstance(channels, int) else channels
        self.kernel_sizes = [kernel_sizes for _ in range(num_blocks + 1)] if isinstance(kernel_sizes, int) else kernel_sizes
        self.catconv_channels = catconv_channels

        self.search_space = search_space
        if search_space is not None:
            self.search_space = TDNNPath(self.space(mode=search_space, channel_ratios=widths, kernels=kernels, depths=depths))
            
        self.layer1 = DynamicConv1dReLUBN(in_feats, self.channels[0], kernel_size=self.kernel_sizes[0])
        self.blocks = nn.ModuleList([
            DynamicSERes2DilatedTDNNBlock(self.channels[0], self.channels[i + 1], self.kernel_sizes[i + 1], dilation) 
                for i, dilation in enumerate(range(2, num_blocks + 2))
        ])
        self.catconv = DynamicConv1d(self.channels[0] * num_blocks, catconv_channels)
        self.pooling = DynamicAttentiveStatsPool(catconv_channels)
        self.bnpool = DynamicBatchNorm1d(catconv_channels * 2)
        self.linear = DynamicLinear(catconv_channels * 2, out_embeds)

        self.bnembed = nn.BatchNorm1d(out_embeds)

        self.active_depth = self.num_blocks
        self.active_channels = self.channels.copy()
        self.active_kernels = self.kernel_sizes.copy()
        self.active_catconv_channels = self.catconv_channels

    def forward(self, x):
        """
            x : Tensor (batch, C, T)
                featuture-first inputs.
        """
        if self.training is True and self.search_space is not None:
            path = self.search_space.sample()
            self.config = path
        
        out = self.layer1(x)
        cat_out = [out]
        sum_out = out
        for i in range(self.active_depth):
            out = self.blocks[i](out) + sum_out
            cat_out.append(out)
            sum_out = sum_out + out 
            # sum_out += out causes inplane operation that raises the RuntimeError while computing gradients
        out = torch.cat(cat_out[1:], dim=1)
        out = F.relu(self.catconv(out))
        out = self.bnpool(self.pooling(out))
        out = self.bnembed(self.linear(out))
        return out

    @property
    def _module_configs(self):
        config_layer1 = (self.in_feats, self.active_channels[0], self.active_kernels[0])
        config_blocks = [(self.active_channels[0], self.active_channels[i + 1], self.active_kernels[i + 1]) for i in range(self.active_depth)]
        cat_indices = torch.arange(self.channels[0] * self.num_blocks, device=self.device).view(self.num_blocks, -1)[:self.active_depth, :self.active_channels[0]].flatten().long()
        config_catconv = (cat_indices, self.active_catconv_channels, None)
        config_pooling = self.active_catconv_channels
        bnpool_indices = torch.arange(self.catconv_channels * 2, device=self.device).view(2,-1)[:, :self.active_catconv_channels].flatten().long()
        config_bnpool = bnpool_indices
        config_linear = (bnpool_indices, None)

        return config_layer1, config_blocks, config_catconv, config_pooling, config_bnpool, config_linear

    @property
    def config(self):
        """
        [TDNN] Config: (depth : int, channels : int/tuple/list, kernels : int/tuple/list, catconv_out_channels : int)

            Examples
            --------
                (3, (256, 256, 256, 256), (5, 3, 3, 3), 768)
        """
        return (self.active_depth, self.active_channels, self.active_kernels, self.active_catconv_channels)

    @config.setter
    def config(self, config):
        if config is None:
            config = (None, None, None, None)
        depth, channels, kernels, catconv_channels = config

        self.active_depth = self.num_blocks if depth is None else depth
        self.active_channels = self.channels.copy() if channels is None else channels
        self.active_kernels = self.kernel_sizes.copy() if kernels is None else kernels
        self.active_catconv_channels = self.catconv_channels if catconv_channels is None else catconv_channels

        config_layer1, config_blocks, config_catconv, config_pooling, config_bnpool, config_linear = self._module_configs

        self.layer1.config = config_layer1
        for i in range(self.active_depth):
            self.blocks[i].config = config_blocks[i]
        self.catconv.config = config_catconv
        self.pooling.config = config_pooling
        self.bnpool.config = config_bnpool
        self.linear.config = config_linear

    def clone(self, config=None):
        self.config = config
        m = DynamicTDNN(self.in_feats, self.out_embeds, self.active_depth, self.active_channels, self.active_kernels)
        m = m.to(self.device)

        config_layer1, config_blocks, config_catconv, config_pooling, config_bnpool, config_linear = self._module_configs

        m.add_module('layer1', self.layer1.clone(config_layer1))
        m.add_module('blocks', nn.ModuleList([
            self.blocks[i].clone(config_blocks[i]) for i in range(self.active_depth)
        ]))
        m.add_module('catconv', self.catconv.clone(config_catconv))
        m.add_module('pooling', self.pooling.clone(config_pooling))
        m.add_module('bnpool', self.bnpool.clone(config_bnpool))
        m.add_module('linear', self.linear.clone(config_linear))
        m.add_module('bnembed', copy.deepcopy(self.bnembed))

        return m.train(self.training)

    def out_size(self, in_size):
        return in_size[:1] + [self.out_embeds]

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)

            Notes
            -----
                MACs are from layer1, (active) blocks, catconv, pooling, bnpool, linear, 
                and bnembed.
        """
        blocks_in_size = [self.layer1.out_size(in_size)]
        for i in range(self.active_depth):
            blocks_in_size.append(self.blocks[i].out_size(blocks_in_size[-1]))
        catconv_in_size = in_size[:1] + [sum([b_in[1] for b_in in blocks_in_size[1:]])] + in_size[2:]
        blocks_in_size = blocks_in_size[:-1]
        pooling_in_size = self.catconv.out_size(catconv_in_size)
        bnpool_in_size = self.pooling.out_size(pooling_in_size)
        linear_in_size = self.bnpool.out_size(bnpool_in_size)
        bnembed_in_size = self.linear.out_size(linear_in_size)

        layer1_ops = self.layer1.count_ops(in_size)
        blocks_ops = []
        for i in range(self.active_depth):
            blocks_ops.append(self.blocks[i].count_ops(blocks_in_size[i]))
        catconv_ops = self.catconv.count_ops(catconv_in_size)
        pooling_ops = self.pooling.count_ops(pooling_in_size)
        bnpool_ops = self.bnpool.count_ops(bnpool_in_size)
        linear_ops = self.linear.count_ops(linear_in_size)
        bnembed_ops = 2 * _prod(bnembed_in_size)

        total_ops = layer1_ops + sum(blocks_ops) + catconv_ops + pooling_ops + bnpool_ops + linear_ops + bnembed_ops
        return total_ops

    def count_params(self):
        layer1_params = self.layer1.count_params()
        blocks_params = []
        for i in range(self.active_depth):
            blocks_params.append(self.blocks[i].count_params())
        catconv_params = self.catconv.count_params()
        pooling_params = self.pooling.count_params()
        bnpool_params = self.bnpool.count_params()
        linear_params = self.linear.count_params()
        bnembed_params = 2.0 * self.out_embeds
        total_params = layer1_params + sum(blocks_params) + catconv_params + pooling_params + bnpool_params + linear_params + bnembed_params
        return total_params

    @property
    def device(self):
        return self.layer1.device

    def set_static(self):
        self._search_space = self.search_space
        self.search_space = None

    def set_dynamic(self):
        self.search_space = self._search_space
        self._search_space = None

    def space(self, mode=['kernel', 'depth', 'width1', 'width2'], channel_ratios=[0.25, 0.35, 0.5, 0.75, 1.0], kernels=[1, 3, 5], depths=[-2, -1, 0]):
        """
        Define the search space of the dynamic network. There are there cagetories:
            kernel: kernel size is dynamic between 5 and 1 in blocks.
            depth: reducing number of blocks is dynamic between -2 and 0 for the whole architecture.
            width: reducing channels of conv layers.

            Notes
            -----
                Once the space is given, the Bound of Params and MACs can be determined. The expected aim would be located at the interval between the lower and upper bounds.
                On the other hand, width is mainly a hyper-parameter for pruning the ECAPA-TDNN architecture.
        """
        
        if 'kernel' in mode: 
            space_kernels = [[ks for ks in kernels if ks <= kernel] for kernel in self.kernel_sizes]
        else:
            space_kernels = [[kernel] for kernel in self.kernel_sizes]
        
        if 'depth' in mode:
            space_depth = [self.num_blocks + d for d in depths]
        else: 
            space_depth = [self.num_blocks]
        if 'width' in mode or 'width1' in mode or 'width2' in mode:
            rates = channel_ratios
            # if 'width1' in mode:
            #     rates = channel_ratios[len(channel_ratios)//2:]
            # if 'width2' in mode or 'width' in mode:
            #     rates = channel_ratios
            space_channels = [[_make_divisible(rate * channel) for rate in rates] for channel in self.channels]
            space_catconv = [_make_divisible(rate * self.catconv_channels) for rate in rates]
        else:
            space_channels = [[channel] for channel in self.channels]
            space_catconv = [self.catconv_channels]
        

        lowbound_depth = min(space_depth)
        lowbound_channels = [min(channels) for channels in space_channels]
        lowbound_kernenls = [min(kernels) for kernels in space_kernels]
        lowbound_catconv = min(space_catconv)

        upbound_depth = max(space_depth)
        upbound_channels = [max(channels) for channels in space_channels]
        upbound_kernenls = [max(kernels) for kernels in space_kernels]
        upbound_catconv = max(space_catconv)
        
        print(f"searchable depth: {space_depth}")
        print(f"searchable channels: {space_channels}")
        print(f"searchable kernels: {space_kernels}")
        print(f"searchable catconv: {space_catconv}")

        return [space_depth] + space_channels + space_kernels + [space_catconv]
    

class TDNNPath:
    def __init__(self, space):
        """
        Parameters
        ----------
            space : list
                The property of space
        """
        import random
        self.nDim = len(space)
        self.space = space
        self.lower = self._config_map([min(choices) for choices in self.space])
        self.upper = self._config_map([max(choices) for choices in self.space])
        self.types = ['discrete' for _ in range(self.nDim)]
        self.g = random.Random(666)
    
    def _config_map(self, config):
        """
        Format config as the configurations of DynamicTDNN.

            Parameters
            ----------
                config : list
        """
        depth = config[0]
        channels = config[1: 1 + (len(config) - 2) // 2]
        kernels = config[1 + (len(config) - 2) // 2: 1 + (len(config) - 2)]
        catconv = config[-1]
        return (depth, channels[: depth + 1], kernels[: depth + 1], catconv)

    def sample(self, dist='random'):
        """Sample a path from search space under the given distribution."""
        assert dist == 'random'
        ans = [self.g.choice(choices) for choices in self.space]
        return self._config_map(ans)

def specfic_tdnn(config):
    """Fixed architecture with initialization of random weights."""
    depth, channels, kernel_sizes, catconv_channels = config
    modelarch = DynamicTDNN(in_feats=80, out_embeds=192, num_blocks=depth, channels=channels, kernel_sizes=kernel_sizes, catconv_channels=catconv_channels, search_space=None, kernels=None, depths=None, widths=None)
    return modelarch


def ecapa_tdnn():
    """ECAPA-TDNN with initialization of random weights."""
    modelarch = DynamicTDNN(in_feats=80, out_embeds=192, num_blocks=3, channels=512, kernel_sizes=[5, 3, 3, 3], catconv_channels=1536, search_space=None, kernels=None, depths=None, widths=None)
    return modelarch


def tdnn8m2g(in_feats=80, out_embeds=192, task='supernet', pretrained=None, accuracy_predictor=None, in_encodec=None, mode_encodec=None, latency_table=None, device='cpu', ks=None, ds=None, ws=None):

    search_space = None if 'supernet' in task else task
    modelarch = DynamicTDNN(in_feats=in_feats, out_embeds=out_embeds, num_blocks=4, channels=512, kernel_sizes=5, catconv_channels=1536, search_space=search_space, kernels=ks, depths=ds, widths=ws).to(device)

    archcodec = None
    acc_pred = None
    tdnnlat = None

    if in_encodec in [48, 408]:
        in_dim = in_encodec
        depth = [2, 3, 4]
        kernel = [1, 3, 5]
        if in_dim == 408:
            width = list(range(128, 513, 8))
            cat = list(range(384, 1537, 8))
        else:
            width = [128, 176, 256, 384, 512]
            cat = [384, 536, 768, 1152, 1536]
        dconfig = {
            'depth': depth, 
            'width0': width, 'width1': width, 'width2': width, 'width3': width, 'width4': width,
            'kernel0': kernel, 'kernel1': kernel, 'kernel2': kernel, 'kernel3': kernel, 'kernel4': kernel, 
            'cat': cat
        }
        archcodec = ArchCoder(12, dconfig, config2d, mode_encodec)

    if accuracy_predictor is not None and os.path.isfile(accuracy_predictor):
        in_dim = int(os.path.basename(accuracy_predictor).split('.')[1])
        assert in_dim == in_encodec or in_dim in [48, 408]
        acc_pred = AccuracyPredictor(in_dim, pretrained=accuracy_predictor, device=device)
    
    if latency_table is not None and os.path.isfile(latency_table):
        tdnnlat = TDNNLatency(latency_table)

    if pretrained is not None and os.path.isfile(pretrained):
        modelweights = torch.load(pretrained, map_location=device)
        modelarch.load_state_dict(modelweights)

    if tdnnlat is not None or acc_pred is not None or in_encodec is not None:
        return modelarch, {'ArchCodec': archcodec, 'AccPredictor': acc_pred, 'LatTable': tdnnlat}

    return modelarch

def tdnn14m4g(in_feats=80, out_embeds=192, task='supernet'):
    modelarch = DynamicTDNN(in_feats=in_feats, out_embeds=out_embeds, num_blocks=4, channels=800, kernel_sizes=5, catconv_channels=1536, search_space = None if 'supernet' in task else task)
    return modelarch

def tdnn16m4g(in_feats=80, out_embeds=192, task='supernet', pretrained=None, ks=None, ds=None, ws=None):
    modelarch = DynamicTDNN(in_feats=in_feats, out_embeds=out_embeds, num_blocks=3, channels=1024, kernel_sizes=5, catconv_channels=1536)
    return modelarch

def tdnn26m7g(in_feats=80, out_embeds=192):
    modelarch = DynamicTDNN(in_feats=in_feats, out_embeds=out_embeds, num_blocks=5, channels=1024, kernel_sizes=5, catconv_channels=1536)
    return modelarch


class ArchCoder(object):
    """
    Encode the configuration of an architecture and Decode it.
    
        Notes
        -----
            This En/DeCoder can be reused.
    """
    def __init__(self, nDim, dconfig: dict, config2d=None, mode=None):
        """
        Parameters
        ----------
            nDim: int
            dconfig: dict
            config2d: function
                Convert the type of config to the dict of it.
        """
        
        self.arch = {}
        self.darch = {}
        self.archd = {}
        self.config2d = config2d if config2d is not None else lambda x: x
        self.nDim = nDim
        self.ndims = []
        self.nfeat = 0
        self.mode = mode # encode mode (None: one-hot, 'b': binary, 'c': continuous)
        assert mode in [None, 'b', 'c']
        
        self._available_arch(dconfig)
        
        for key, value in dconfig.items():
            value = sorted(list(set(value)))
            self.arch[key] = value
            self.ndims.append(key)
            if self.mode is None:
                self.nfeat = self.nfeat + len(value)
            elif self.mode == 'b':
                self.nfeat = self.nfeat + len(format(len(value), 'b'))
            elif self.mode == 'c':
                self.nfeat = self.nfeat + 1
            self.darch[key] = {}
            self.archd[key] = {}
            for i, v in enumerate(value):
                self.darch[key][v] = i
                self.archd[key][i] = v
                
    def _available_arch(self, dconfig):
        assert self.nDim == len(dconfig.keys())
        
    def encode(self, config: list):
        """Encode subnet to the specific form, e.g., one-hot"""
        dconfig = self.config2d(config)
        
        self._available_arch(dconfig)
        
        one_hot = np.zeros(self.nfeat, dtype=np.float32)
        
        nstart = 0
        for key in self.ndims:
            
            if self.mode is None:
                next_idx = nstart + len(self.darch[key])
            elif self.mode == 'b':
                next_idx = nstart + len(format(len(self.darch[key]), 'b'))
            elif self.mode == 'c':
                next_idx = nstart + 1

            v = dconfig[key]

            if v is None:
                pass
            else:
                assert v in self.darch[key], 'v: {} not in darch[{}]'.format(v, key)
                i = self.darch[key][v]
                if self.mode is None:
                    one_hot[nstart + i] = 1.0
                elif self.mode == 'b':
                    i = format(i + 1, 'b').zfill(len(format(len(self.darch[key]), 'b')))
                    for j, ii in enumerate(i):
                        one_hot[nstart + j] = int(ii)
                elif self.mode == 'c':
                    one_hot[nstart] = (i + 1) * 1.0 / len(self.darch[key])
            nstart = next_idx
        return one_hot.astype(np.float32)
        
    def decode(self, feat: np.ndarray, typ: int = 0):
        assert self.nfeat == feat.shape[0]
        assert typ in [0, 1]
        typ = {0: 'dict', 1: 'list'}[typ]
        
        feat = feat.astype(np.int) if self.mode is not 'c' else feat
        dconfig = {}
        
        nstart = 0
        for key in self.ndims:
            if self.mode is None:
                next_idx = nstart + len(self.darch[key])
                part_feat = feat[nstart : next_idx]
                i = np.where(part_feat == 1)[0]
                i = None if len(i) != 1 else i.item()
                v = None if i is None else self.archd[key][i]
            elif self.mode == 'b':
                next_idx = nstart + len(format(len(self.darch[key]), 'b'))
                part_feat = feat[nstart : next_idx]
                i = int(''.join([str(i) for i in part_feat]), 2)
                v = None if i == 0 else self.archd[key][i - 1]
            elif self.mode == 'c':
                next_idx = nstart + 1
                i = feat[nstart : next_idx].item()
                v = None if i == 0 else self.archd[key][round(i * len(self.darch[key])) - 1]
            dconfig[key] = v
            nstart = next_idx
        
        if typ == 'dict': return dconfig
        
        config = [dconfig[key] for key in self.ndims if dconfig[key] is not None]
        return config


def config2d(config):
    """
    Convert the list of configuration to a format of dict. It is used as the `config2d` of `ArchCoder`.
    """
    assert len(config) == 1 + 2 + config[0] * 2 + 1
    dconfig = {
        'depth': None, 
        'width0': None, 'width1': None, 'width2': None, 'width3': None, 'width4': None,
        'kernel0': None, 'kernel1': None, 'kernel2': None, 'kernel3': None, 'kernel4': None, 
        'cat': None
    }
    dconfig['depth'] = config[0]
    for i, w in enumerate(config[1: config[0] + 2]):
        dconfig['width%d' % i] = w
    for i, k in enumerate(config[config[0] + 2: 2 * config[0] + 3]):
        dconfig['kernel%d' % i] = k
    dconfig['cat'] = config[-1]
    return dconfig


class AccuracyPredictor(nn.Module):
    """
    Build a predictor that predicts accuarcy with respect to the configiuration of architectures.

        Notes
        -----
            The predictor is a general model that predicts accuracy based on one-hot inputs.
            Hence, the accuracy predictor can be reused.
    """

    def __init__(self, in_feats, pretrained=None, device='cpu', transform=False, verbose=False):
        """
        Parameters
        ---------
            in_feats : int
        """
        super(AccuracyPredictor, self).__init__()
        self.in_feats = in_feats
        self.model = nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        ).to(device)
        self.ynorm = nn.BatchNorm1d(1).to(device)
        self.device = device
        self.transform = transform
        self.verbose = verbose
        self.fit_process = {}
        
        if pretrained:
            if os.path.exists(pretrained):
                state_dict = torch.load(pretrained, map_location=device)
                self.model.load_state_dict(state_dict['model'])
                self.ynorm.load_state_dict(state_dict['ynorm'])
            else:
                raise FileNotFoundError(pretrained)
        else:
            if verbose: print('MUST train the prodictor FIRST!')
            
    def forward(self, inputs, label=None):
        out = self.inputs_transform(inputs)
        out = self.model(inputs)
        return out

    def inputs_transform(self, inputs):
        if self.transform:
            return 2.0 / (1 + torch.exp(-4.0 * inputs)) - 1.0
        return inputs

    @torch.no_grad()
    def predict(self, inputs):
        """
        Predict via the given sub-network config.

            Parameter
            ---------
                inputs : torch.Tensor or numpy.ndarray

            Return
            ------
                y_pred : numpy.ndarray
                    1-dimensional vector of predicted accuracy of the given resnet's architecture
        """
        assert isinstance(inputs, (np.ndarray, torch.Tensor))
        assert inputs.shape[-1] == self.in_feats
        
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        inputs = inputs.to(self.device)
        
        self.eval()    
        y_pred = self.forward(inputs)
        # inv batchnorm1d
        bn = self.ynorm
        y_pred = (y_pred - bn.bias) / (bn.weight + bn.eps) * torch.sqrt(bn.running_var + bn.eps) + bn.running_mean 
        y_pred = y_pred.detach().squeeze().cpu().numpy()
        
        return y_pred

    def fit(self, X, y, n_epoch=50, lr=1e-3, ce='mse', optimizer='adam', scheduler='steplr', alpha=1e-3, gamma=0.5, cycle=2):
        """
        Train the model that predicts accuracy of the given sub-network's config.
        
            Parameter
            ---------
                X : numpy.ndarray
                y : numpy.ndarray
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[-1] == self.in_feats
        assert ce in ['mse', 'l1', 'smoothl1']
        assert optimizer in ['sgd', 'adam']
        assert scheduler in ['steplr', 'cycliclr']

        X_copy = X.copy()
        y_copy = y.copy()
        
        from torch.utils.data import TensorDataset, DataLoader
        
        X = torch.FloatTensor(X)
        X = X.to(self.device)
            
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = torch.FloatTensor(y)
        y = y.to(self.device)
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
        
        if ce == 'mse':
            criterion = torch.nn.MSELoss().to(self.device)
        elif ce == 'l1':
            criterion = torch.nn.L1Loss().to(self.device)
        else:
            criterion = torch.nn.SmoothL1Loss().to(self.device)

        if not self.fit_process:
            if optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=alpha)
            else:
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=alpha)
            if scheduler == 'steplr':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, n_epoch // cycle, gamma=gamma)
            else:
                cycle_momentum = not isinstance(optimizer, torch.optim.Adam)
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, lr * gamma, lr, step_size_up=(y.shape[0] // loader.batch_size) * (n_epoch // 2 // cycle), mode='triangular2', cycle_momentum=cycle_momentum)
        else:
            optimizer = self.fit_process['optimizer']
            scheduler = self.fit_process['scheduler']
        
        import sys
        if any('jupyter' in arg for arg in sys.argv):
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        
        self.train()
        pbar = tqdm(range(n_epoch), total=n_epoch)
        for epoch in pbar:
            
            nloss = 0
            for inp, lab in loader:
                optimizer.zero_grad()
                outp = self.forward(inp, lab)
                nlab = self.ynorm(lab)
                loss = criterion(outp, nlab)
                loss.backward()
                optimizer.step()
                
                iloss = loss.item()
                nloss += iloss

                if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                    scheduler.step()
                
            nloss /= y.shape[0] // loader.batch_size
            pbar.set_description('Epoch %03d' % (epoch + 1))
            pbar.set_postfix({'Loss': nloss, 'LR': scheduler.get_last_lr()[0]})

            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR) and epoch < n_epoch - 1: 
                scheduler.step()
        
        self.fit_process['optimizer'] = optimizer
        self.fit_process['scheduler'] = scheduler

        y_pred = self.predict(X_copy)
        l1error = np.abs(y_pred - y_copy).mean()
        print("Train Epoch %d at L1 Error %.3f and LR %.6f" % (n_epoch, l1error, scheduler.get_last_lr()[0]))

    def save(self, model_path):
        """
        Save the trained model.
        
            Examples
            --------
                >>> predictor.save('/workspace/projects/sugar/works/nas/exps/exp3/acc1/acc.pth.tar')
        """
        if os.path.exists(model_path):
            raise FileExistsError(model_path)
        else:
            state_dict = {'model': self.model.state_dict(), 'ynorm': self.ynorm.state_dict()}
            torch.save(state_dict, model_path)


class TDNNLatency(object):
    """
    Approximate Latency Table for Dynamic TDNN. It support:

        1. return fixed layer latency: embedding batch normalization.
        2. return dynamic layer latency: layer1, blocks, catconv, pool, and embedding.
        
        Notes
        -----
            - The operation of cat vary with respect to in/out channels of blocks.
            - In order to fasten the process, some operations are unestimated or ignored. For example, 
            do not estimate `cat`, `bnpool`, and `bnembed`, ignore the difference of dilation between 
            TDNN `blocks`.

        Details
        -------
            The configuration of units are follows:

            - layer1: Conv1d + ReLU + BN with various output channels
                - output: 128:512:8 (lower/upper/step)
            - (ignore different dilation) blocks: TDNN Block with various input/output/internal channels 
                and different internal kernel sizes
                - input/output: 128:512:8
                - internal: 
                    - channels: 128:512:8
                    - kernel sizes: 1, 3, 5
            - (unestimated) cat: cat operation with various input channels
                - input: 2/3/4 x 128:512:8
            - catconv: Conv1d + ReLU with various input/output channels
                - input: cat(2/3/4 x 128:512:8)
                - output: 384:1536:8
            - pool: AttentiveStatsPool with various input/output sizes
                - input/output: 384:1536:8
            - (unestimated) bnpool: BN with various input/output sizes
                - input/output: 384:1536:8
            - embed: linear with various input sizes
                - input: 384:1536:8
            - (fixed and unestimated) bnembed: BN with the fixed sizes
    """
    def __init__(self, table = None):
        """
            Parameters
            ----------
                table : dict
                    directory to a latency table
        """
        if table is not None:
            if not os.path.exists(table):
                raise FileExistsError(table)
            with open(table, 'r') as f:
                self.latency_table = yaml.safe_load(f)
        else:
            self.latency_table = {}
            print('Warning: no latency table is loaded. Please measure latency first!')

    def query(self, config):
        """Return latency with the given configuration of a sub-architecture."""
        total_time = 0
        depth, channel, kernel, catconv = config
        param = {
            'layer1': {
                'in_channels': 80, 
                'out_channels': channel[0],
                'kernel_size': kernel[0]
            },
            'blocks': [
                {
                    'channels': channel[0], 
                    'mid_channels': channel[1 + i], 
                    'kernel_size': kernel[1 + i], 
                } 
                for i in range(depth)
            ],
            'catconv': {
                'in_channels': channel[0] * depth, 
                'out_channels': catconv
            },
            'pool': {
                'in_dim': catconv
            },
            'embed': {
                'in_features': catconv * 2,
                'out_features': 192
            },
        }
        for key, value in param.items():
            if not isinstance(value, list):
                value = [value]
            total_time += sum([self.latency_table[key][self.map_param(v)] for v in value])
        return total_time

    def measure(self, model_class, device='cpu', exclude=['dilation']):
        """
        Measure latency of Supernet ResNet via 4-second inputs.

            Parameters
            ----------
                model_class : dict
                    A dictionary of Module Class with various configurations. For example,

                    {
                        'layer1': {
                            'class': DynamicConv1dReLUBN, 
                            'kwargs': {
                                'in_channels': [80], 
                                'out_channels': list(range(128, 513, 8)), 
                                'kernel_size': [1, 3, 5]}
                            },
                            'input': [[80, 300]]
                        },
                        'blocks': { ... }
                        'catconv': { ... },
                        'pool': { ... },
                        'embed': { ... },
                    }

                device : str
                    Device for deployment, such as 'cpu' or 'cuda:0'

            Note
            ----
                meausure in cpu input 1 sample, but in cuda, input 1 batch of 64 samples.
        """
        assert sorted(list(model_class.keys())) == sorted(['layer1', 'blocks', 'catconv', 'pool', 'embed'])

        num_measure = 0
        for value in model_class.values():
            num_measure += _prod([len(vi) for vi in value['kwargs'].values()])
        print('The number of measure is {:,} totally.'.format(num_measure))

        from sklearn.model_selection import ParameterGrid
        from sugar.metrics import latency
        import sys
        if any('jupyter' in arg for arg in sys.argv):
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.latency_table = {}
        for key, value in model_class.items():
            self.latency_table[key] = {}
            param_grid = value['kwargs']
            param_list = list(ParameterGrid(param_grid))
            for param in tqdm(param_list, total=len(param_list), desc='Measure {}'.format(key)):
                m = value['class'](**param)
                for in_size in value['input']:
                    in_size = [1] + in_size if device == 'cpu' else [64] + in_size
                    in_size = [param[s] if isinstance(s, str) else s for s in in_size] # s in param
                    m_lat = latency(m, in_size, device=device)
                    m_param = self.map_param({key: value for key, value in param.items() if key not in exclude})
                    self.latency_table[key][m_param] = m_lat

    def map_param(self, param : dict):
        keys = sorted(list(param.keys()))
        return '({})'.format(', '.join('{}={}'.format(k, str(param[k])) for k in keys))

    def to_yaml(self, yaml_file):
        if not os.path.exists(yaml_file):
            with open(yaml_file, 'w') as f:
                yaml.dump(self.latency_table, f)
        else:
            raise FileExistsError(yaml_file)


class MinimizeError(ea.Problem):
    """Interface to Minimize Error Rate for tdnn8m2g."""
    def __init__(self, arch_codec, acc_pred, lat_table, space_mode='Coarse', threshold=50, name='Test'):
        """[reusable]"""
        assert space_mode in ['Coarse', 'Fine']
        M = 1  # the number of objective(s)
        maxormins = [1]  # initilize maxormins (min: 1, max: -1)
        Dim = 12  # initilize the dimension of varibales
        varTypes = [1] * Dim  # initilize the type of variables (Continuous: 0, Discrete: 1)
        
        if space_mode == 'Coarse':
            ## Coarse-grained Space
            # depth: {2, 3, 4},
            # channel: {128, 176, 256, 384, 512},
            # kernel: {1, 3, 5},
            # catconv: {384, 536, 768, 1152, 1536}
            lb = [2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  # variables' lower bound
            ub = [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4]  # variables' upper bound
        else:
            ## Fine-grained Space
            # channel: [0 - 48] * 8 + 128
            # kernel: {1: 1, 2: 3, 3: 5}
            # catconv: [0, 144] * 8 + 384
            lb = [2,  0,  0,  0,  0,  0, 1, 1, 1, 1, 1,   0]  # variables' lower bound
            ub = [4, 48, 48, 48, 48, 48, 3, 3, 3, 3, 3, 144]  # variables' upper bound
        
        lbin = [1] * Dim  # include or exclude lower bound (include: 1, exclude: 0)
        ubin = [1] * Dim  # include or exclude upper bound (include: 1, exclude: 0)
        
        # instantiate problem
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        
        # tools for objective
        self.space_mode = space_mode
        self.arch_codec = arch_codec
        self.acc_pred = acc_pred
        self.lat_table = lat_table
        self.threshold = threshold

    def aimFunc(self, pop):
        """[implemented]Compute objective(s) function with contraints."""
        X = pop.Phen.copy()  # obtain variables
        X = self.encode(X)
        
        X_onehot = np.array([self.arch_codec.encode(x) for x in X])
        ObjV = self.acc_pred.predict(X_onehot)
        ObjV = np.stack([ObjV], 1)
        pop.ObjV = ObjV  # compute objective(s) as ObjV
        
        pop.CV = np.zeros((pop.sizes, 1))
        exidx = self.constraints(X)
        pop.CV[exidx, 0] = 1 # mark the individuals that not meet constraints
        
    def getReferObjV(self, reCalculate=False):
        referenceObjV = None
        self.ReferObjV = referenceObjV
        return referenceObjV
        
    def constraints(self, subnets):
        raise NotImplementedError
        
    def encode(self, X):
        """[reusable]Convert solutions to the encode of subnets."""
        ndim = X.ndim
        if ndim == 1:
            X = X[np.newaxis, :]
        
        if self.space_mode == 'Coarse':
            channel_dict = {0:128, 1:176, 2:256, 3:384, 4:512}
            catconv_dict = {0:384, 1:536, 2:768, 3:1152, 4:1536}
            # channel: {128, 176, 256, 384, 512}
            X[:, 1:6] = np.array([[channel_dict[xi] for xi in x] for x in X[:, 1:6]])
            # kernel: {1, 3, 5}
            X[:, 6:11] = X[:, 6:11] * 2 - 1
            # catconv: {384, 536, 768, 1152, 1536}
            X[:, 11] = np.array([catconv_dict[x] for x in X[:, 11]])
        else:
            X[:, 1:6] = X[:, 1:6] * 8 + 128 # channel: [0 - 48] * 8 + 128
            X[:, 6:11] = X[:, 6:11] * 2 - 1 # kernel: {1: 1, 2: 3, 3: 5}
            X[:, 11] = X[:, 11] * 8 + 384   # catconv: [0, 144] * 8 + 384

        X = X.tolist()
        X = [[x[0]] + x[1: 2+x[0]] + x[6: 7+x[0]] + [x[-1]] for x in X]
        if ndim == 1:
            X = X[0]
        return X

def ea_solve(target, problem, encode='BG', NIND=50, MAXGEN=800, Pm=0.1, verbose=True):
    """
    Solve search problem using evolutionary algorithm.
        target = 'EER'
        latency_threshold = 15 # ms 8 ~
        space_mode = 'Coarse' # coarse or fine
        problem = MinEER(arch_codec, acc_pred, lat_table, space_mode=space_mode, threshold=latency_threshold)
    """
    ## Set population
    Encoding = encode  # Encode manner
    NIND = NIND  # Population size
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Condition descriptor
    population = ea.Population(Encoding, Field, NIND)  # Instantiate Population
    ## Set algorithm
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # Instantiate algorithm
    myAlgorithm.MAXGEN = MAXGEN  # Maximal evolutionary generations
    myAlgorithm.mutOper.Pm = Pm  # Mutation probility
    myAlgorithm.logTras = 50  # Interval for log (No log: 0)
    myAlgorithm.verbose = verbose  # Determine whether to print log
    myAlgorithm.drawing = 0  # Plot fasion (No plot: 0, Result plot: 1, objective plot: 2, solution plot: 3)
    ## Evolutionary Process
    [BestIndi, population] = myAlgorithm.run()  # Perform algorithm
    ## Print result
    if verbose:
        print('The number of evaluation: %s' % myAlgorithm.evalsNum)
        print('It took %.3f s' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        
        phen = problem.encode(BestIndi.Phen[0])
        subnet = (phen[0], phen[1: 2+phen[0]], phen[2+phen[0]: 3+2*phen[0]], phen[-1])
        lat = problem.lat_table.query(subnet) if problem.lat_table else -1.0
        if verbose:
            print('The Optimal %s: %.2f%%' % (target, BestIndi.ObjV[0][0]))
            print('The optimal solution: {}: {:.1f}ms'.format(subnet, lat))
        return subnet, {'ACC': BestIndi.ObjV[0][0], 'Latency': lat}
    else:
        if verbose: print('No solutions!')
        return None

if __name__ == '__main__':
    
    device = 'cpu'

    # net = tdnn6m2g(in_feats=80, out_embeds=192).to(device)
    # print(net.__class__.__name__, macs_params(net, verbose=False, device=device))
    # netpath = TDNNPath(net)
    # print('Low:', netpath.lower, macs_params(net.clone(netpath.lower), verbose=False, device=device))
    # print('Up:', netpath.upper, macs_params(net.clone(netpath.upper), verbose=False, device=device))

    net = tdnn16m4g() # tdnn8m2g(in_feats=80, out_embeds=192).to(device)

    print(net.count_ops([1, 80, 300]))

    print(net.__class__.__name__, macs_params(net, verbose=False, device=device))

    modes = ['kernel', 'depth', 'width1', 'width2']
    channel_ratios = [0.25, 0.35, 0.5, 0.75, 1.0]
    kernel_sizes = [3, 5]
    depths = [-2, -1, 0]
    netpath = TDNNPath(net.space(mode=modes, channel_ratios=channel_ratios, kernels=kernel_sizes, depths=depths))
    
    print(netpath.space)

    print('Low:', macs_params(net.clone(netpath.lower), custom_ops=None, verbose=False, device=device), netpath.lower)
    print('Up:', macs_params(net.clone(netpath.upper), custom_ops=None, verbose=False, device=device), netpath.upper)

    # net = tdnn14m4g(in_feats=80, out_embeds=192).to(device)
    # print(net.__class__.__name__, macs_params(net, verbose=False, device=device))
    # netpath = TDNNPath(net)
    # print('Low:', netpath.lower, macs_params(net.clone(netpath.lower), verbose=False, device=device))
    # print('Up:', netpath.upper, macs_params(net.clone(netpath.upper), verbose=False, device=device))

    # net = tdnn26m7g(in_feats=80, out_embeds=192).to(device)
    # print(net.__class__.__name__, macs_params(net, verbose=False, device=device))
    # netpath = TDNNPath(net)
    # print('Low:', netpath.lower)
    # print(macs_params(net.clone(netpath.lower), verbose=False, device=device))
    # print('Up:', netpath.upper)
    # print(macs_params(net.clone(netpath.upper), verbose=False, device=device))

    # netpath = TDNNPath(net.space(mode=modes))

    # for _ in range(10):
    #     path = netpath.sample()
    #     cloned = net.clone(path)
    #     print(macs_params(cloned, verbose=False, device=device), 'Path:', path)
    
    print(1)
