"""
Efficient Architecture for Speaker Recognition in the Wild

1) Once for All Network:
    This work borrows from the once-for-all network in [1] for 
    finding out the optimal sub-network that makes the resulted 
    architecture more efficient and maintains the performance.
2) Teacher-Student Framework:
    This work applies the teacher-student framework for network
    compression so that the resulted architecture will be small 
    and effective.

The search of the optimal sub-network originates from the work of 
    once-for-all in github [2].

[1] Cai, H., Gan, C., Wang, T., Zhang, Z., Han, S., 2020. 
    Once-for-All: Train One Network and Specialize it for 
    Efficient Deployment, in: International Conference 
    on Learning Representations.

[2] OFA. https://github.com/mit-han-lab/once-for-all


================================================================
Algorithm 1. Efficient Speaker Embedding Learning
----------------------------------------------------------------
Requirement:
    a model family
    a trained teacher network
Return: 
    a trained network as a sub-network
----------------------------------------------------------------
1.  Define a supernet from the given model family
2.  Initilize the supernet
3.  Training supernet:
4.      Update the weights of the supernet via Algorithm 2
5.  Searching archtecture:
6.      Select a specialized sub-network via Algorithm 3
7.  Return the selected sub-network 
================================================================


================================================================
Algorithm 2. Supernet Training Method
----------------------------------------------------------------
Requirement:
    a model
    a optimizer
    model inputs
    model outputs
Return: 
    a trained model

Note:
    参考论文 Once-for-All: Train One Network and Specialize it 
        for Efficient Deployment 中的 progressive shrinking 
        algorithm (渐进收缩)
----------------------------------------------------------------
1.  3-second supernet training
2.  6-second 1 single path one-shot method to kernel size (卷积核)
3.  6-second 2 single path one-shot method to depth (网络层数)
4.  6-second 4 single path one-shot method to width (通道数量)
5.  Return the trained model
================================================================


================================================================
Algorithm 3. Predictor-based Sub-network Evoluaionary Search Method
----------------------------------------------------------------
Requirement:
    a model family
    evaluation dataset
    evaluation metrics
    model constrains
Return: 
    a model as sub-network from the given model
----------------------------------------------------------------
1.  Extract hyper-parameters from the model family
2.  Define metric predictors in terms of the hyper-parameters
3.  Initilize the predictors
4.  Train the predictors
5.  Searching hyper-parameters:
6.      Update solutions via evolution search
7.  Return the found model from the optimal hyper-parameters
================================================================

Model:
1) resnet
    - supernet: resnet
    - search space: frame length, kernel size (e), network depth (d),
        channel width(w).
        - channel (ConvBNReLU)
        - kernel (ConvBNReLU)
        - depth (ResNet)
    - training strategy: progressive training
    - architecture search: evolutionary search
    - implementation: ConvBNReLU -> L2/3BasicBlock -> ResNet
        - elastic properties is specilized. 
            1. It always return maximum value that has been set to.
            2. It is always modified via its active value.
            3. The value returned can be different from that to be 
                set to.

Dataset:
1) VoxCeleb1: speaker verification, speaker identification
"""
import copy
import random

import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

__all__ = ['ResNet', 'ResNetPath']

def _count_parameters(net, verbose=0):
    if verbose > 0:
        for name, p in net.named_parameters():
            if p.requires_grad:
                print(name, p.numel())
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params

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
    Product of list of sizes. It is faster than numpy.prod and torch.prod.
        
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


class StatsPool(nn.Module):
    '''
    Source: https://github.com/cvqluu/Factorized-TDNN
    '''

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def count_ops(self, in_size):
        """
        Count MACs/MAdd operations of this module.
        """
        B, F, T = in_size
        out_size = (B, F * 2)
        total_ops = B * F * (4 * T - 1)
        return total_ops, out_size

    def forward(self, x):
        '''
        Compute mean and unbiased (default:false) std. 
            input: size (batch, input_features, seq_len)
            outpu: size (batch, output_features)
        '''
        means = torch.mean(x, dim=2) # (batch, F)
        t = x.shape[2]
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(2)
        numerator = torch.sum(residuals**2, dim=2)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def count_ops(self, in_size):
        """
        Count MACs/MAdd operations of this module.
        """
        out_size = in_size
        total_ops = _prod(in_size)
        return total_ops, out_size

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hsigmoid()'


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.in_channels = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Linear(channel, channel // reduction)
        self.relu = nn.ReLU(inplace=True)  # ReLU 会造成梯度消失
        self.expand = nn.Linear(channel // reduction, channel)
        self.hsigmoid = Hsigmoid()
        self.reduction = reduction

    def clone(self, in_channels):
        m = SELayer(in_channels, self.reduction)
        m = m.to(self.reduce.weight.device)
        mid_channels = in_channels // self.reduction
        m.reduce.weight.data.copy_(
            self.reduce.weight[:mid_channels, :in_channels].data
        )  # reduce weight
        m.reduce.bias.data.copy_(
            self.reduce.bias[:mid_channels].data
        )  # reduce bias
        m.expand.weight.data.copy_(
            self.expand.weight[:in_channels, :mid_channels].data
        )  # expand weight
        m.expand.bias.data.copy_(
            self.expand.bias[:in_channels].data
        )  # expand bias
        return m.train(self.training)

    def count_ops(self, in_size):
        """
        Count MACs/MAdd operations of this module. And it support variable channels of inputs.
        """
        out_size = in_size
        total_ops = 0
        # ada_avg_pool
        kernel_ops = _prod(in_size[2:]) + 1
        total_ops += kernel_ops * (in_size[0] * in_size[1])
        # reduce
        total_ops += in_size[1] * (in_size[0] * (in_size[1] // self.reduction))
        # relu
        total_ops += 0
        # expand
        total_ops += (in_size[1] // self.reduction) * (in_size[0] * in_size[1])
        # hsigmoid
        total_ops += self.hsigmoid.count_ops(in_size[:2])[0]
        # x * y
        total_ops += _prod(in_size) # This is a place that profile can not count.
        return total_ops, out_size

    def reindex(self, indices):
        """
        Reindex weights via indices.

            Parameter
            ---------
                indices : IntTensor or LongTensor
                    the 1-D tensor containing the indices to index
        """
        # in and out reindex
        self.reduce.weight.data = torch.index_select(self.reduce.weight.data, 1, indices) # reduce in_features
        self.expand.weight.data = torch.index_select(self.expand.weight.data, 0, indices) # expand out_features
        self.expand.bias.data   = torch.index_select(self.expand.bias.data, 0, indices)   # expand out_features
        # middle reindex
        importance = torch.sum(torch.abs(self.expand.weight.data), dim=(0,))
        _, re_idx = torch.sort(importance, dim=0, descending=True)
        self.expand.weight.data = torch.index_select(self.expand.weight.data, 1, re_idx) # expand in_features
        self.reduce.weight.data = torch.index_select(self.reduce.weight.data, 0, re_idx) # reduce out_features
        self.reduce.bias.data   = torch.index_select(self.reduce.bias.data, 0, re_idx)   # reduce out_features

    def forward(self, x):
        """Support variable channels of inputs."""
        b, in_channels, _, _ = x.size()
        y = self.avg_pool(x).view(b, in_channels)
        if in_channels != self.in_channels:
            mid_channels = in_channels // self.reduction
            y = F.linear(y, self.reduce.weight[:mid_channels, :in_channels], self.reduce.bias[:mid_channels])
            y = self.relu(y)
            y = F.linear(y, self.expand.weight[:in_channels, :mid_channels], self.expand.bias[:in_channels])
        else:
            y = self.reduce(y)
            y = self.relu(y)
            y = self.expand(y)
        y = self.hsigmoid(y)
        y = y.view(b, in_channels, 1, 1)
        return x * y


class ConvBNReLU(nn.Module):
    """
    The unit is core. It consists of Conv2D + BatchNorm + ReLU + SE, and its variable parameters are expansion and kernel.
        The unit can be illustrated as:

               # =============== #
              # =  in_channels  = #
             #                     #
            # ===  in transfrom === #
            # = (if in_transform) = #
            # --------------------- #
            #                       #
            #        Conv2d         #
            #                       #
            # == kernel transform = #
            # == (if use_kernel) == #
            #                       #
            # --------------------- #
            # === out transfrom === #
            # =(if out_transform) = #
            # --------------------- #
             #                     #
              #                   #
               # = out_channels= #
               # =============== #
               # =  BatchNorm  = #
               # =============== #
               # =     ReLU    = #
               # =============== #
               # =    SELayer  = #
               # =============== #

        Compared to the traditional ConvBNReLU layer, it adds 2 linear transform (in transform and kernel transform).
        Also, the Conv2d, BatchNorm, and SELayer is adjustable of dimensions of inputs and output. 
        Those perform:
            1) active_depth = True: 
                elastic depth, for example, apply in transform to in_channel dimension of Conv2d.
            2) active_channels < out_channels:
                elastic width, for example, use the first k of out_channel dimension of Conv2d, BatchNorm, SELayer.
            3) use_kernel = True and active_kernel < kernel_size: 
                elastic kernel, for example, apply kernel transform to kernel of Conv2d.
    """
    BASE_KERNEL = [1, 3, 5, 7]
    BASE_EXPAND = [0.25, 0.5, 1.0]


    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, 
                 use_norm=False, use_relu=False, use_se=False, 
                 use_kernel=False, elastic_depth=False, in_transform=False, out_transform=False):
        """
        The unit can include Conv2d, BatchNorm2d, ReLU, and SELayer. It support 
            elastic channels of output and elastic kernel size.

            Parameter
            ---------
                use_kernel : bool
                    it determine to support elastic kernel size.
                elastic_depth : bool
                    it determine to support linear transform for elastic depth.
                in_transform : bool
                    it determine to linearly transform the conv input channels.
                out_transform : bool
                    it determine to linearly transform the conv output channels.
            Note
            ----
                use_kernel is to center out the kernel's weights and then linearly transform it.
                elastic_depth is to support elastic depth.
                in/out transform is to linearly transform the input/output channels, which is used to support alternative elastic width.
        """
        assert not (elastic_depth is True and in_transform is True)

        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.padding       = padding
        self.dilation      = dilation
        self.groups        = groups
        self.use_norm      = use_norm
        self.use_relu      = use_relu
        self.use_se        = use_se
        self.use_kernel    = use_kernel
        self.elastic_depth = elastic_depth
        self.in_transform  = in_transform
        self.out_transform = out_transform

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation=dilation, groups=groups, bias=False)
        if use_norm is True:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu is True:
            self.relu = nn.ReLU(inplace=True)
        if use_se is True:
            self.se = SELayer(out_channels)

        self.active_channels = self.out_channels  # Determine the output
        self.active_kernel   = self.kernel_size
        
        # Support 7 to 5, 5 to 3, and 3 to 1.
        if use_kernel is True:  
            if self.kernel_size > 5:
                self.k75 = nn.Parameter(torch.eye(25), True)
            if self.kernel_size > 3:
                self.k53 = nn.Parameter(torch.eye(9), True)
            if self.kernel_size > 1:
                self.k31 = nn.Parameter(torch.eye(1), True)

        # support alternative in elastic depth
        # Perform as conv2d with kernel size of 1
        if elastic_depth is True: 
            self.active_depth  = 0
            self.one_inchannel = nn.Parameter(torch.eye(in_channels).view(in_channels, in_channels, 1, 1))
            self.two_inchannel = nn.Parameter(torch.eye(in_channels).view(in_channels, in_channels, 1, 1))

        # Linearly transform the input (fixed) channel: 1 to 1/2 and 1/2 to 1/4
        # Perform as conv2d with kernel size of 1
        if in_transform is True: 
            half_size    = _make_divisible(in_channels * 0.5)
            quarter_size = _make_divisible(in_channels * 0.25)
            self.half_inchannel = nn.Parameter(torch.eye(half_size, half_size).view(half_size, half_size, 1, 1))
            self.quarter_inchannel = nn.Parameter(torch.eye(quarter_size, quarter_size).view(quarter_size, quarter_size, 1, 1))

        # Linearly transform the output (variable) channel: 1 to 1/2 and 1/2 to 1/4
        if out_transform is True: 
            half_size    = _make_divisible(out_channels * 0.5)
            quarter_size = _make_divisible(out_channels * 0.25)
            self.half_outchannel    = nn.Parameter(torch.eye(half_size, half_size))
            self.quarter_outchannel = nn.Parameter(torch.eye(quarter_size, quarter_size))

    def clone(self, in_channels, out_channels, kernel_size=None):
        """Clone the module and convert the convolution layer to be without kernel/in/out transformation."""
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        m = ConvBNReLU(in_channels, out_channels, kernel_size, stride=self.stride, dilation=self.dilation, 
                       groups=self.groups, use_norm=self.use_norm, use_relu=self.use_relu, use_se=self.use_se, 
                       use_kernel=False, elastic_depth=False, in_transform=False, out_transform=False)
        m = m.to(self.conv.weight.device)
        # copy and transform conv2d
        with torch.no_grad():
            weights = self.kernel_forward(in_channels, out_channels, kernel_size)
        m.conv.weight.data.copy_(weights.data)
        # copy bn
        if self.use_norm is True:

            m.bn.weight.data.copy_(self.bn.weight[:out_channels].data)  # bn weight
            m.bn.bias.data.copy_(self.bn.bias[:out_channels].data)  # bn bias

            # copy statistics of batchnorm is meaningless, because the batchnorm's statistics depends on
            #   architecture before it. Thus, we have to recompute them after cloning the model. 
            # The natural way to recompute the batchnorm's statistics is to perform forward inference of
            #   the cloned model on a part of dataset, which can be from training set.
            m.bn.running_mean.data.copy_(self.bn.running_mean[:out_channels].data)  # bn running mean
            m.bn.running_var.data.copy_(self.bn.running_var[:out_channels].data)  # bn running var

            m.bn.eps = self.bn.eps
            m.bn.momentum = self.bn.momentum
            m.bn.track_running_stats = self.bn.track_running_stats

        # copy se
        if self.use_se:
            m.se = self.se.clone(out_channels)
        return m.train(self.training)

    @property
    def channels(self):
        return self.in_channels, self.out_channels

    @channels.setter
    def channels(self, out_channels):
        assert out_channels <= self.out_channels, "%d > max channels %d" % (
            out_channels, self.out_channels)
        self.active_channels = out_channels

    @property
    def kernel(self):
        return self.kernel_size

    @kernel.setter
    def kernel(self, kernel_size):
        """Support kernel size of 7, 5, 3, 1."""
        assert kernel_size in {7, 5, 3, 1, 0}, "Invalid kernel size %d != 7/5/3/1/0" % kernel_size
        self.active_kernel = kernel_size if kernel_size != 0 else self.kernel
    
    @property
    def depth(self):
        return self.elastic_depth

    @depth.setter
    def depth(self, active_level):
        """Determine to linearly transform the input channels"""
        assert self.elastic_depth is True
        self.active_depth = active_level

    def kernel_forward(self, in_planes, out_planes, out_kernel):
        """
        [Core]Support elastic kernel:
            1) variable kernel weights: reduce size to a smaller
            2) variable in channels: weight in channels
            3) variable out channels: weight out channels
            4) especially, variable in channels for elastic depth
            5) speed up in practical.

            Note 
            ----
                `Tensor[start_id : end_ed]` may result in warnings in PyTorch 1.6, and `Tensor[start_id : end_ed].contiguous()` does work. The warning is as:` [W accumulate_grad.h:165] Warning: grad and param do not obey the gradient layout contract. This is not an error, but may impair performance.` The warnings is not error.

                To perform faster, traspose/permute + linear operations are replaced by conv2d to in channel and by torch.mm to out channel, in practical.
        """
        weights = self.conv.weight
        out_channels, in_channels = weights.size()[0:2]

        # elastic kernel size
        # To clear, torch.matmul replace torch.nn.functional.linear
        if self.use_kernel is True: # once-for-all mode
            if self.kernel > 5 and out_kernel < 7:  # 7 to 5
                # weights = F.linear(weights[:, :, 1:-1, 1:-1].contiguous().view(
                #     out_channels, in_channels, 25), self.k75).view(out_channels, in_channels, 5, 5)
                weights = torch.matmul(weights[:, :, 1:-1, 1:-1].contiguous().view(
                    out_channels, in_channels, 25), self.k75).view(out_channels, in_channels, 5, 5)
            if self.kernel > 3 and out_kernel < 5:  # 5 to 3
                # weights = F.linear(weights[:, :, 1:-1, 1:-1].contiguous().view(
                #     out_channels, in_channels, 9), self.k53).view(out_channels, in_channels, 3, 3)
                weights = torch.matmul(weights[:, :, 1:-1, 1:-1].contiguous().view(
                    out_channels, in_channels, 9), self.k53).view(out_channels, in_channels, 3, 3)
            if self.kernel > 1 and out_kernel < 3:  # 3 to 1
                # weights = F.linear(weights[:, :, 1:-1, 1:-1].contiguous().view(
                #     out_channels, in_channels, 1), self.k31).view(out_channels, in_channels, 1, 1)
                weights = torch.matmul(weights[:, :, 1:-1, 1:-1].contiguous().view(
                    out_channels, in_channels, 1, 1), self.k31).view(out_channels, in_channels, 1, 1)
        elif out_kernel < self.kernel: # straight mode
            idx = (self.kernel - out_kernel) // 2
            weights = weights[:, :, idx: -idx].contiguous()

        # is active depth
        # Note that conv2d is practically faster than transpose + linear, around 2 times.
        # On the other hand, in speed it's equal to torch.mm(inchannel, weights.transpose(0, 1).reshape(?)).reshape(?').transpose(0,1)), which is not easily readable.
        if self.elastic_depth is True:
            if self.active_depth >= 1: # last not top block forward
                # weights = F.linear(weights.transpose(1, 3), self.one_inchannel).transpose(1, 3)
                weights = F.conv2d(weights, self.one_inchannel)
            if self.active_depth >= 2:
                # weights = F.linear(weights.transpose(1, 3), self.two_inchannel).transpose(1, 3)
                weights = F.conv2d(weights, self.two_inchannel)

        # elastic in channels
        # Note that conv2d is practically faster than transpose + linear, around 2 times.
        # On the other hand, in speed it's equal to torch.mm(inchannel, weights.transpose(0, 1).reshape(?)).reshape(?').transpose(0,1)), which is not easily readable.
        if self.in_transform is True: # alternative mode
            half_inplanes = _make_divisible(self.in_channels * 0.5)
            quarter_inplanes = _make_divisible(self.in_channels * 0.25)
            if in_planes < self.in_channels: # 1 to 1/2
                weights = F.conv2d(weights[:, :half_inplanes].contiguous(), self.half_inchannel)
            if in_planes < half_inplanes and quarter_inplanes < half_inplanes: # 1/2 to 1/4, ensure quarter_outplaces != half_outplanes == 8
                weights = F.conv2d(weights[:, :quarter_inplanes].contiguous(), self.quarter_inchannel)
        else: # straight mode
            weights = weights[:, :in_planes].contiguous()
        
        # elastic out channels
        # Note that torch.mm is practically faster than transpose + linear.
        if self.out_transform is True: # alternative mode
            half_outplanes = _make_divisible(self.out_channels * 0.5)
            quarter_outplaces = _make_divisible(self.out_channels * 0.25)
            _, s1, s2, s3 = weights.size()
            if out_planes < self.out_channels: # 1 to 1/2
                weights = torch.mm(self.half_outchannel, weights[:half_outplanes].reshape(half_outplanes, s1 * s2 * s3).reshape(half_outplanes, s1, s2, s3))
            if out_planes < half_outplanes and quarter_outplaces < half_outplanes: # 1/2 to 1/4, ensure quarter_outplaces != half_outplanes == 8
                weights = torch.mm(self.quarter_outchannel, weights[:quarter_outplaces].contiguous().reshape(quarter_outplaces, s1 * s2 * s3)).reshape(quarter_outplaces, s1, s2, s3)
        else: # straight mode
            weights = weights[:out_planes].contiguous()

        return weights

    def conv_forward(self, x):
        """Pass first #out_channels filters."""
        in_channels = x.size()[1]

        # ensure in and out channels and kernel size are original
        if in_channels == self.in_channels and self.active_channels == self.out_channels and self.active_kernel == self.kernel_size and (
            self.elastic_depth is False or (self.elastic_depth is True and self.active_depth == 0)):
            return self.conv(x)
    
        out_channels = self.active_channels
        out_kernel = self.active_kernel
        weights = self.kernel_forward(in_channels, out_channels, out_kernel)
        padding = (out_kernel - 1) // 2
        return F.conv2d(x, weights, bias=None,
                        stride=self.stride, padding=padding,
                        dilation=self.dilation, groups=1)

    def bn_forward(self, x):
        """
        Pass first #out_channels filters. This function is modified from the source code of _BatchNorm in Pytorch.
        """
        if x.size()[1] == self.out_channels:
            return self.bn(x)

        if self.bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.bn.momentum
        if self.bn.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked = self.bn.num_batches_tracked + 1
                if self.bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.bn.momentum
        out_channels = self.active_channels
        return F.batch_norm(x,
                            self.bn.running_mean[:out_channels],
                            self.bn.running_var[:out_channels],
                            self.bn.weight[:out_channels],
                            self.bn.bias[:out_channels],
                            self.bn.training or not self.bn.track_running_stats,
                            exponential_average_factor, self.bn.eps)

    def count_ops(self, in_size):
        """
        Count MACs/MAdd operations of this module.
        """
        def after_conv(c, dil=1, ks=3, stride=1):
            pad = (ks - 1) // 2
            c = (c + 2 * pad - dil * (ks - 1) - 1) / stride + 1
            return int(c)
        
        H_out = after_conv(in_size[2], dil=self.dilation, ks=self.active_kernel, stride=self.stride)
        W_out = after_conv(in_size[3], dil=self.dilation, ks=self.active_kernel, stride=self.stride)
        out_size = (in_size[0], self.active_channels, H_out, W_out)
        total_ops = 0
        # conv
        kernel_ops = self.active_kernel * self.active_kernel * (in_size[1] // self.groups)
        total_ops += kernel_ops * _prod(out_size)
        total_ops += sum([ks * ks for ks in range(self.kernel_size - 2, self.active_kernel - 1, -2)])
        # bn
        if self.use_norm:
            total_ops += 2 * _prod(out_size)
        # relu
        if self.use_relu:
            total_ops += 0
        # se
        if self.use_se:
            ops, out = self.se.count_ops(out_size)
            total_ops += ops
            assert out == out_size
        return total_ops, out_size

    def reindex(self, indices, dim=0):
        """
        Reindex weights via indices. There are two mode:
            1) dim = 1: reindex to input of conv
            2) dim = 0: reindex to output of conv

            Parameter
            ---------
                indices : IntTensor or LongTensor
                    the 1-D tensor containing the indices to index
        """
        self.conv.weight.data = torch.index_select(self.conv.weight.data, dim, indices)
        if dim == 0:
            if self.use_norm:
                self.bn.weight.data = torch.index_select(self.bn.weight.data, 0, indices)
                self.bn.bias.data   = torch.index_select(self.bn.bias.data, 0, indices)
                if type(self.bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
                    self.bn.running_mean.data = torch.index_select(self.bn.running_mean.data, 0, indices)
                    self.bn.running_var.data  = torch.index_select(self.bn.running_var.data, 0, indices)
            if self.use_se:
                self.se.reindex(indices)

    def forward(self, x):
        x = self.conv_forward(x)
        if self.use_norm:
            x = self.bn_forward(x)
        if self.use_relu:
            x = self.relu(x)
        if self.use_se:
            x = self.se(x)
        return x


class L3BasicBlock(nn.Module):
    """
    3-Level basic block of ResNet. It is core unit of elastic ResNet.
        The mode can be illustrated as:

            # ====== (fixed) in planes ====== #
             #                               #
              #         3x3                 #
               #        kernel             #
                #       size              #
                 #                       #
                  # ==== expansion ==== #
                  #    (0.25/0.5/1.0)   #
                  #                     #
                  #     1/3/5/7x        #
                  #     active          #
                  #     kernel          #
                  #     size            #
                  #                     #
                  #    (0.25/0.5/1.0)   #
                  # ==== expansion ==== #
                 #                       #
                #       1x1               #
               #        kernel             #
              #         size                #
             #                               #
            # ====== (fixed) out planes ====== #

        Variable parameters: expansion/expand ratio and kernel size.
        To support various channels, channels sort via importance is achieved as Once-for-All. (unfinished)
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expand_ratio=1.0, 
                 downsample=None, use_se=False, elastic_depth=False):
        """
        L3 BasicBlock include 3 subblocks, where the 2nd subblock can be of
            elastic kernel size. Expand ratio is also elasitc to the output
            of the 1st subblock, the input and output of 2nd subblock, and the
            input of 3rd subblock.
            Note that the input and output of the block is fixed.

            Parameter
            ---------
                expand_ratio : float
                    Support elastic expand ratio.
                kernel_size : int
                    Support elastic kernel size.
        """
        super(L3BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.elastic_depth = elastic_depth
        mid_planes = _make_divisible(in_planes * expand_ratio)
        self.subblock1 = ConvBNReLU(in_planes, mid_planes, 3, stride, use_norm=True, use_relu=True, elastic_depth=elastic_depth)
        self.subblock2 = ConvBNReLU(
            mid_planes, mid_planes, kernel_size, 1, use_norm=True, use_relu=True, use_se=use_se, use_kernel=True, in_transform=True
        )
        self.subblock3 = ConvBNReLU(mid_planes, out_planes, 1, 1, use_norm=True, in_transform=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.active_expand_ratio = expand_ratio
        self.active_out_channels = out_planes # reversed to subblock3

    def clone(self, expand_ratio, kernel_size, in_channel=None, out_channel=None):

        in_channel = self.in_planes if in_channel is None else in_channel
        mid_channels = _make_divisible(in_channel * expand_ratio)
        out_channel = self.active_out_channels if out_channel is None else out_channel

        m = L3BasicBlock(in_channel, out_channel, kernel_size, stride=self.stride, expand_ratio=expand_ratio, 
                         downsample=self.downsample, use_se=self.use_se)
        m = m.to(self.subblock1.conv.weight.device)
        m.add_module('subblock1', self.subblock1.clone(in_channel, mid_channels))
        m.add_module('subblock2', self.subblock2.clone(mid_channels, mid_channels, kernel_size))
        m.add_module('subblock3', self.subblock3.clone(mid_channels, out_channel))

        if self.downsample is not None:
            m.add_module('downsample', self.downsample.clone(in_channel, out_channel))
        return m.train(self.training)

    @property
    def channels(self):
        return self.in_planes, self.out_planes

    @channels.setter
    def channels(self, out_channels):
        """Currently, it is not considered."""
        self.active_out_channels = out_channels

    @property
    def expansion(self):
        return self.expand_ratio

    @expansion.setter
    def expansion(self, expand_ratio):
        """Change the output channels of subblock 1 and 2."""
        self.active_expand_ratio = expand_ratio

    @property
    def kernel(self):
        return self.subblock2.kernel

    @kernel.setter
    def kernel(self, kernel_size):
        """Change the kernel size of subblock 2."""
        assert self.subblock2.use_kernel is True, "subblock2 must be use_kernel True"
        self.subblock2.kernel = kernel_size

    def _update_config(self, x_size):
        """Update active config."""
        in_channal = x_size[1]
        mid_channel = _make_divisible(in_channal * self.active_expand_ratio)
        self.subblock1.channels = mid_channel
        self.subblock2.channels = mid_channel
        self.subblock3.channels = self.active_out_channels

    def count_ops(self, in_size):
        """
        Count MACs/MAdd operations of this module.
        """
        self._update_config(in_size)

        total_ops = 0
        # subblock1
        ops1, out1 = self.subblock1.count_ops(in_size)
        total_ops += ops1
        # subblock2
        ops2, out2 = self.subblock2.count_ops(out1)
        total_ops += ops2
        out_size = out2
        # subblock3
        ops3, out3 = self.subblock3.count_ops(out2)
        total_ops += ops3
        out_size = out3
        # downsample
        if self.downsample is not None:
            ops_ds, out_ds = self.downsample.count_ops(in_size)
            total_ops += ops_ds
            assert out_ds == out_size
        # += identity
        total_ops += _prod(out_size) # This is a place that profile can not count.
        # relu
        total_ops += 0
        return total_ops, out_size

    def reindex(self):
        """
        Reindex weights via importance.
        """
        ## subblock2 -> 1
        importance = torch.sum(torch.abs(self.subblock2.conv.weight.data), dim=(0, 2, 3))
        _, re_idx  = torch.sort(importance, dim=0, descending=True)
        # reindex subblock1's output
        self.subblock1.reindex(re_idx, dim=0)
        # reindex subblock2's input
        self.subblock2.reindex(re_idx, dim=1)

        ## subblock3 -> 2
        importance = torch.sum(torch.abs(self.subblock3.conv.weight.data), dim=(0, 2, 3))
        _, re_idx  = torch.sort(importance, dim=0, descending=True)
        # reindex subblock2's output
        self.subblock2.reindex(re_idx, dim=0)
        # reindex subblock3's input
        self.subblock3.reindex(re_idx, dim=1)

    def forward(self, x):

        self._update_config(x.size())
        
        identity = x
        out = self.subblock1(x)
        out = self.subblock2(out)
        out = self.subblock3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Elastic Neural Network for Speaker Recognition.
        Model Family: MagNetO Modified ResNet-34
        It consist 4 parts:
            1) the first convolution filter (input_stem), 
                such as conv2d(128, 3, 1, 1) + nb + relu
            2) the next multiple blocks (blocks),
            3) the pooling layer (pool),
            4) the fully-connected layer.
        Variable parameters: depth, width, and kernel.
    """
    BASE_DEPTH = [3, 4, 6, 3]
    BASE_WIDTH = [128, 128, 256, 256]
    BASE_SE = [False, False, True, True]
    BASE_KERNEL = [3, 3, 3, 3]
    EXPANSION = 1

    def __init__(self,
                 depths=[3, 4, 6, 3],
                 widths=[128, 128, 128, 128, 128, 128, 128, 256,
                         256, 256, 256, 256, 256, 256, 256, 256],
                 kernels=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
                 pather=None,
                 in_feats=80, 
                 out_embeds=256):
        """
        Model Familly: ResNet.
            It is modified from:
                Garcia-Romero, 2020. MagNetO: X-vector Magnitude Estimation Network plus Offset 
                for Improved Speaker Recognition. https://doi.org/10.21437/odyssey.2020-1

            Parameter
            ---------
                depths: list
                    The layers in each block. The ResNet has 4 elastic blocks.
                widths: list
                    The channels in each layers. The ResNet has 16 elastic layers.
                kernels: list
                    The kernels' size in each layers corresponding to elastic layers.
                pather: ResNetPath
                    The path selector is used to choose a path in forward process. It
                    is usually dynamic and controlled by hyperparameters of architectures.
                in_feats: int
                    The number of input's feature related to the frequency dimension of 
                    temporal pooling layer. Specifically, the output's demension of the 
                    pooling layer is one out of four of `in_feats`.

            Example
            -------
                Large Static ResNet as MagNetO, where `pather` is None. It works for
                    VoxCeleb2 dataset which has more than 1,000,000 utterances.
                    ```
                    arch = ResNet(depths=[3, 4, 6, 3],
                                  widths=[128, 128, 128, 128, 128, 128, 128, 256,
                                          256, 256, 256, 256, 256, 256, 256, 256],
                                  kernels=[5, 5, 5, 5, 5, 5, 5, 5, 
                                           5, 5, 5, 5, 5, 5, 5, 5],
                                  pather=None)
                    ```
                Small Elastic ResNet as [1], where `pather` is created. It works for
                    VoxCeleb1 dataset which has around 130,000 utterances.
                    ```
                    selector = ResNetPath(depths=[3, 4, 6, 3],
                                          widths=[0.25, 0.5, 1.0],
                                          kernels=[1, 3],
                                          depth_minus=2)
                    arch = ResNet(depths=[3, 4, 6, 3],
                                  widths=[16, 16, 16, 32, 32,  32,  32,  64,
                                          64, 64, 64, 64, 64, 128, 128, 128],
                                  kernels=[5, 5, 5, 5, 5, 5, 5, 5, 
                                           5, 5, 5, 5, 5, 5, 5, 5],
                                  pather=selector,
                                  in_feats=64)
                    ```
                [1] Cai, W., 2018. Exploring the Encoding Layer and Loss Function in End-to-End Speaker 
                    and Language Recognition System, Odyssey 2018. pp. 74–81. https://doi.org/10.21437/odyssey.2018-11.
                    where the optimizer is as SGD with momentum 0.9 and weight decay 1e-4. The learning rate is from
                    0.1 -> 0.01 -> 0.001.
        """
        widths  = [w for w in widths if w is not None and w > 0]
        kernels = [k for k in kernels if k is not None and k > 0]
        assert sum(depths) == len(widths) == len(kernels), '%d ?= %d ?= %d' % (
            sum(depths), len(widths), len(kernels))
        super(ResNet, self).__init__()
        self.in_feats = in_feats # It relates to the dimension of the temporal pooling layer
        self.out_embeds = out_embeds  # It relates to the size of embeddings
        self.depth_list = depths
        self.width_list = [_make_divisible(w) for w in widths]
        self.kernel_list = kernels
        self.pather = pather
        n1_conv2d = widths[0]
        self.conv = ConvBNReLU(1, n1_conv2d, 3, 1, use_norm=True, use_relu=True)
        blocks = []
        # ResBlock-1 x3
        blocks += [L3BasicBlock(n1_conv2d, widths[0], kernels[0])]
        blocks += [L3BasicBlock(widths[i-1], widths[i], kernels[i]) for i in range(1, depths[0])]
        # ResBlock-2a
        idx = depths[0]
        blocks += [L3BasicBlock(widths[idx-1], widths[idx], kernels[idx], 2, elastic_depth=True,
                                downsample=ConvBNReLU(widths[idx-1], widths[idx], stride=2, use_norm=True))]
        # ResBlock-2b x3
        blocks += [L3BasicBlock(widths[i-1], widths[i], kernels[i])
                   for i in range(depths[0] + 1, sum(depths[:2]))]
        # ResBlock-3a
        idx = sum(depths[:2])
        blocks += [L3BasicBlock(widths[idx-1], widths[idx], kernels[idx], 2, use_se=True, elastic_depth=True,
                                downsample=ConvBNReLU(widths[idx-1], widths[idx], stride=2, use_norm=True))]
        # ResBlock-3b x5
        blocks += [L3BasicBlock(widths[i-1], widths[i], kernels[i], use_se=True)
                   for i in range(sum(depths[:2]) + 1, sum(depths[:3]))]
        # ResBlock-4a
        idx = sum(depths[:3])
        blocks += [L3BasicBlock(widths[idx-1], widths[idx], kernels[idx], 2, use_se=True, elastic_depth=True,
                                downsample=ConvBNReLU(widths[idx-1], widths[idx], stride=2, use_norm=True))]
        # ResBlock-4b x2
        blocks += [L3BasicBlock(widths[i-1], widths[i], kernels[i], use_se=True)
                   for i in range(sum(depths[:3]) + 1, sum(depths[:4]))]
        # resnet has 4 groups of blocks
        self.blocks_indices = [
            [i for i in range(depths[0])],
            [depths[0] + i for i in range(depths[1])],
            [sum(depths[0:2]) + i for i in range(depths[2])],
            [sum(depths[0:3]) + i for i in range(depths[3])]
        ]
        self.blocks = nn.Sequential(*blocks)
        self.pool = StatsPool()
        # 4 = in_feats / (2 * 2 * 2) * 2: 3 times stride 2 downsample and a statistic pooling
        self.fc = nn.Linear(widths[-1] * self.in_feats // 4, self.out_embeds)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.active_depth = self.depth_list.copy()
        self.active_width = self.width_list.copy()
        self.active_kernel = self.kernel_list.copy()

    def __repr__(self):
        return 'Params: %.2fG' % (_count_parameters(self) / 1e6)

    @property
    def depth(self):
        return self.depth_list.copy()

    @depth.setter
    def depth(self, depths):
        """
        Change the depth of ResNet.

            Example
            -------
                self.depths = [3, 4, 6, 4]
        """
        assert len(depths) == len(
            self.depth_list), 'Invalid Length %d != %d' % (len(depths), len(self.depth))
        self.active_depth = depths.copy()
        self.blocks[3].subblock1.depth = self.depth_list[0] - depths[0]
        self.blocks[7].subblock1.depth = self.depth_list[1] - depths[1]
        self.blocks[13].subblock1.depth = (self.depth_list[2] - depths[2] + 1) // 2

    @property
    def width(self):
        return self.width_list.copy()

    @width.setter
    def width(self, magnifications):
        """
        Parameter
        ---------
            magnifications: A list of float
                A list of the times of magnification that magnify/minify the 
                original channels.

        Example
        -------
            self.width = [0.25, 0.5, 1.0]
        """
        assert len(magnifications) == len(
            self.blocks), 'Invalid Length %d != %d' % (len(magnifications), len(self.blocks))
        for expand_ratio, block in zip(magnifications, self.blocks):
            block.expansion = expand_ratio
        self.active_width = [_make_divisible(
            mag * w) for mag, w in zip(magnifications, self.width_list)]

    @property
    def kernel(self):
        return self.kernel_list.copy()

    @kernel.setter
    def kernel(self, kernel_sizes):
        """
        Example
        -------
            self.kernel = [1, 3, 5, 7]
        """
        assert len(kernel_sizes) == len(
            self.blocks), 'Invalid Length %d != 16' % len(kernel_sizes)
        for kernel_size, block in zip(kernel_sizes, self.blocks):
            block.kernel = kernel_size
        self.active_kernel = kernel_sizes.copy()

    @property
    def config(self):
        """Return architecture config"""
        return self.depth + self.width + self.kernel

    @property
    def active_config(self):
        """Return active arcitecture config"""
        return self.active_depth + self.active_width + self.active_kernel

    def subnet(self,
               depths: list = None,
               expand_ratios: list = None,
               kernel_sizes: list = None):
        """
        Activate a sub-network from the given hyper-parameter. Note that `None` means the total size.

            Parameter
            ---------
                depths: list of 4 int
                    The depth of 4 parts. For example, [3, 4, 6, 3]
                expand_ratios: list of 16 float.
                    The ratio to the original width. For example, 
                    [1.0 for _ in range(16)]
                kernel_sizes: list of 16 int
                    The kernel of each block. For example, 
                    [5 for _ in range(16)]
        """
        if depths is not None:
            self.depth = depths
        else:
            self.depth = self.depth

        if depths is not None:
            indices = [] # indicate not available modules
            for d1, d0 in zip(depths, self.depth):
                assert d1 <= d0
                part_indices = [1 for _ in range(d1)] + [0 for _ in range(d1, d0)]
                indices.extend(part_indices)
        else:
            indices = [1 for _ in range(sum(self.depth))]

        if expand_ratios is not None:
            self.width = expand_ratios
        else:
            self.width = self.width
        # convert invalid width to None as depth
        self.active_width = [mag if idx == 1 else None for idx,
                             mag in zip(indices, self.active_width)]
        
        if kernel_sizes is not None:
            self.kernel = kernel_sizes
        else:
            self.kernel = self.kernel
        # convert invalid kernel size to None as depth
        self.active_kernel = [ks if idx == 1 else None for idx,
                              ks in zip(indices, self.active_kernel)]

    def clone(self, depths: list, expand_ratios: list, kernel_sizes: list):
        """
        Clone a sub-network for the given hyper-parameter.

            Parameter
            ---------
                depths : list
                    The depths of 4 blocks. It is smaller than [3, 4, 6, 3].
                expand_ratios : list
                    The expand ratios of all layers in all blocks, usually lower than 1.0.
                kernel_sizes : list
                    The kernel sizes of all layers in all blocks, usually 1, 3, 5, 7.

            Return
            ------
                m : ResNet
                    A copy of active sub-network.

            Note that clone from a device (e.g., CUDA or cpu) to a device should be careful. For example, 
            copy data in the same device which keep numberic data equal, otherwise, may result in extremely
            sight error as copy from GTX 1080Ti to CPU. However, it may not degrade the accuracy of a model.
        """
        self.depth = depths # set the active_depth of input channels of 3/7/13 subblocks that is used for one/two input channels
        
        indices = [] # indicate not available modules
        for d1, d0 in zip(depths, self.depth):
            assert d1 <= d0
            part_indices = [1 for _ in range(d1)] + [0 for _ in range(d1, d0)]
            indices.extend(part_indices)
        # remain blocks indices corresponding to depth
        indices = [idx for idx, flag in enumerate(indices) if flag == 1]

        widths = [_make_divisible(self.width_list[idx] * expand_ratios[idx]) for idx in indices]
        kernel_sizes = [kernel_sizes[idx] for idx in indices]
        expand_ratios = [expand_ratios[idx] for idx in indices]
        
        m = ResNet(depths=depths, widths=widths, kernels=kernel_sizes, in_feats=self.in_feats, out_embeds=self.out_embeds)
        m = m.to(self.fc.weight.device)

        # replace the original conv: delete and then add
        m.add_module('conv', self.conv.clone(self.conv.in_channels, self.conv.out_channels))
        # blocks
        blocks = []
        for idx, mag, ks in zip(indices, expand_ratios, kernel_sizes):
            blocks.append(self.blocks[idx].clone(mag, ks))
        # replace the original blocks: delete and then add
        m.add_module('blocks', nn.Sequential(*blocks))
        # replace the original fc: delete and then add
        fc = nn.Linear(self.fc.in_features, self.fc.out_features)
        fc = fc.to(self.fc.weight.device)
        fc.weight.data.copy_(self.fc.weight.data)
        fc.bias.data.copy_(self.fc.bias.data)
        m.add_module('fc', fc)
        return m.train(self.training)

    def count_memory(self, in_size=[1, 1, 64, 400], verbose=True):
        """
        Query MACs/MAdd and Parameters by model and input size. It do not require latency table.
        """
        params = self.count_params()
        macs   = self.count_ops(in_size)
        if verbose:
            macs, params = thop.clever_format([macs, params], "%.2f")
        return macs, params

    def count_params(self):
        """Count parameters of this module."""
        n = 0
        for param in self.parameters():
            if param.requires_grad:
                n += param.numel()
        return n

    def count_ops(self, in_size):
        """
        Count MACs/MAdd operations of this module.
        """
        total_ops = 0
        # conv
        ops_conv, out_conv = self.conv.count_ops(in_size)
        total_ops += ops_conv
        # blocks
        in_block = out_conv
        for depth, indices in zip(self.active_depth, self.blocks_indices):
            for idx in indices[:depth]:
                ops_block, in_block = self.blocks[idx].count_ops(in_block)
                total_ops += ops_block
        # pool
        out_block = (in_block[0], in_block[1] * in_block[2], in_block[3])
        ops_pool, out_pool = self.pool.count_ops(out_block)
        total_ops += ops_pool
        # fc
        out_size = (out_pool[0], self.fc.out_features)
        ops_fc = out_pool[1] * (out_size[0] * out_size[1])
        total_ops += ops_fc
        # relu
        total_ops += 0
        return total_ops

    def reorganize(self):
        """
        Reorganize weights of elastic blocks via importance. After it, each block uses the most important
            channels as initial weights, and each subblock of linear uses the most important weights as 
            initial weights.

            Note that it is suitable to retrained network.
        """
        for idx in range(len(self.blocks)):
            self.blocks[idx].reindex()

    def forward(self, x):
        if self.training: # select a dynamic path while training
            if self.pather is not None:  
                sub_depth, sub_width, sub_kernel = self.pather.depth_width_kernel
                # activate a specific sub-network
                self.subnet(depths=sub_depth, expand_ratios=sub_width, kernel_sizes=sub_kernel)
        x = self.conv(x)
        for depth, indices in zip(self.active_depth, self.blocks_indices):
            for idx in indices[:depth]:
                x = self.blocks[idx](x)
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        x = self.pool(x)
        x = self.fc(x)
        return x


class ResNetPath(object):
    """
    Selection forward paths for the given constraints in a specific 
        elastic network.
        It support discrete variables.

        Example
        -------
            Example 1: 
                selector = ResNetPath([3, 4, 6, 3], [3, 5, 7], [0.25, 0.5, 1.0], 0)
                Create 
                    (1) boarder [3] [4] [6] [3] in depth
                    (2) boarder [3, 5, 7] in kernel
                    (3) boarder [0.25, 0.5, 1.0] in width

            Example 2:
                selector = ResNetPath([3, 4, 6, 3], [3, 5, 7], [0.5, 1.0], 1)
                Create 
                    (1) boarder [2,3] [3,4] [4,5,6] [2,3] in depth
                    (2) boarder [3, 5, 7] in kernel
                    (3) boarder [0.5, 1.0] in width

            Example 3:
                selector = ResNetPath([3, 4, 6, 3], [5, 7], [0.25, 0.5, 1.0], 2)
                Create 
                    (1) boarder [1,2,3] [2,3,4] [2,3,4,5,6] [1,2,3] in depth
                    (2) boarder [5, 7] in kernel
                    (3) boarder [0.25, 0.5, 1.0] in width
    """

    def __init__(self, 
                 depths=[3, 4, 6, 3],
                 widths=[0.25, 0.5, 1.0],
                 kernels=[1, 3, 5],
                 depth_minus=2, 
                 strategy='random', 
                 probability=None):
        """
        Parameter
        ---------
            depths : list
                the maximal number of layers in each block
            widths : list
                the candidate width ratio
            kernels : list
                the candidate kernel size
            depth_minus : int
                the number of the first layers skipped
            strategy : str
                Support random or defined selection.
        """
        self.name = "Elastic ResNet"
        self.nDepth = depths.copy()
        self.nKernel = kernels.copy()
        self.nWidth = widths.copy()
        self.nNode = sum(depths)
        # Dimensional of variables: depth, width, and kernel, where depth 2 is twice minus, 
        #   specifically, 
        #   1) depth_minus = 0 has original bound [3, 4, 6, 3];
        #   2) depth_minus = 1 has lower depth bound [2, 3, 4, 2]; 
        #   3) depth_minus = 2 has lower depth bound [1, 2, 2, 1]. 
        self.nDim = len(depths) + sum(depths) + sum(depths)
        self.borders = [list(range(d - depth_minus * (1 + int(i == 2)), d + 1)) for i, d in enumerate(depths)] + [
            widths for _ in range(sum(depths))] + [kernels for _ in range(sum(depths))]
        self.borders_choice = sum([len(border) for border in self.borders])
        self.borders_dict = [{b: i for i, b in enumerate(border)} for border in self.borders]
        self.variables = ['depth'] * len(depths) + ['width'] * sum(depths) + ['kernel'] * sum(depths)
        self.strategy = strategy
        self.probability = probability

    def __repr__(self):
        """Fomulate the candidates of sub-networks."""
        repr = ''
        repr += self.name + '\n'
        repr += 'Number of Nodes: %d\n' % self.nNode
        repr += 'Number of Variables: %d\n' % self.nDim
        repr += 'Border of Variables:\n'
        for v, b in zip(self.variables, self.borders):
            repr += '\t%s: ' % v
            if v == 'depth' or v == 'kernel':
                repr += ' '.join(['%d' % bi for bi in b])
            else:
                repr += ' '.join(['%.2f' % bi for bi in b])
            repr += '\n'
        return repr

    def _random(self, verbose=False):
        """Randomly select an architecture."""
        v = []
        for b in self.borders:
            v.append(random.choice(b))
        if verbose is True:
            for vname, vi, bi in zip(self.variables, v, self.borders):
                print(vname, vi, '\tin', bi)
        return v

    def _defined(self, verbose=False):
        """Lowerly index selected with higher probability."""
        v = []
        if self.probability is None:
            return self._random(False)

        for i, b in enumerate(self.borders):
            v.append(random.choices(b, weights=self.probability[i])[0])

        if verbose is True:
            for vname, vi, bi in zip(self.variables, v, self.borders):
                print(vname, vi, '\tin', bi)
        return v

    def select(self):
        if self.strategy == 'random':
            return self._random(verbose=False)
        elif self.strategy == 'defined':
            return self._defined(verbose=False)

    def _none2zero(self, candidate : list):
        """
        Convert some expand ratio and kernel size that is not in sub-network to zero.
        
            Parameter
            ---------
                candidate : list or numpy.ndarray
                    A candidate corresponding to a sub-network's config.
        """

        if not isinstance(candidate, list):
            candidate = list(candidate)

        sub_depth = candidate[0:4].copy()
        sub_expand_ratio = candidate[4:20].copy()
        sub_kernel_size = candidate[20:36].copy()

        indices = [] # indicate not available modules
        for d1, d0 in zip(sub_depth, self.nDepth):
            assert d1 <= d0
            part_indices = [1 for _ in range(d1)] + [0 for _ in range(d1, d0)]
            indices.extend(part_indices)

        for idx, flag in enumerate(indices):
            if flag == 0:
                sub_expand_ratio[idx] = 0
                sub_kernel_size[idx]  = 0

        return sub_depth + sub_expand_ratio + sub_kernel_size

    def config2onehot(self, candidate : list):
        """
        Convert a list of config to onehot list.

            Parameter
            ---------
                candidate : list or list[list[,]]
            
            Return
            ------
                onehot : list or list[list[,]]
        """
        assert isinstance(candidate, list)

        if isinstance(candidate[0], list): # list[list[,]]
            return [self.config2onehot(c) for c in candidate]

        # convert invalid config to zero
        candidate = self._none2zero(candidate)
        # initialize onehot
        onehot = [0 for _ in range(self.borders_choice)]
        # indicate valid config in the onehot
        start_idx = 0
        for idx, c in enumerate(candidate):
            if c != 0:
                num = self.borders_dict[idx][c]
                onehot[start_idx + num] = 1
            start_idx += len(self.borders[idx])
        return onehot

    @property
    def depth_width_kernel(self):
        """Return a pair of depth, expand ratio, and kernel size of sub-networks."""
        selected_vars = self.select()
        sub_depth = selected_vars[0:4]
        sub_expand_ratio = selected_vars[4:20]
        sub_kernel_size = selected_vars[20:36]
        return sub_depth, sub_expand_ratio, sub_kernel_size


class ResNetLatency(object):
    """
    Latency Table for Elastic ResNet. It support:
        1) return fixed layer latency: conv, pool, and fc
        2) return elastic layer latency: blocks[0-15]
        where reshape is costless and is not considered.
    """
    def __init__(self, table: str = '', depths=[3, 4, 6, 4], expand_ratios=[0.25, 0.5, 1.0], kernel_size=[1, 3, 5]):
        """
            Parameter
            ---------
                table : dict
                    directory to a latency table

            Note
            ----
                the loaded latency table is as follows.
                    The average cost of time (ms) in each layer of ResNet.
                    {
                        'conv'  : float,
                        'blocks' : [
                            { '%.2f-%d' % (expand_ratio, kernel_size): float},
                            { '%.2f-%d' % (expand_ratio, kernel_size): float},
                            ...
                        ],
                        'pool'  : float,
                        'fc'    : float,
                    }
        """
        import yaml
        if table:
            with open(table, 'r') as f:
                self.latency_table = yaml.safe_load(f)
        else:
            self.latency_table = {}
            print('Warning: no latency table is loaded. Please measure latency first!')

        self.depths        = depths
        self.expand_ratios = expand_ratios
        self.kernel_sizes  = kernel_size

    def query(self, depths: list, expand_ratios: list, kernel_sizes: list):
        """Return latency with the given config without input size"""
        indices = [] # indicate not available modules
        for d1, d0 in zip(depths, self.depths):
            assert d1 <= d0
            part_indices = [1 for _ in range(d1)] + [0 for _ in range(d1, d0)]
            indices.extend(part_indices)
        # remain blocks indices corresponding to depth
        indices = [idx for idx, flag in enumerate(indices) if flag == 1]

        fixed_time   = self.latency_table['conv'] + self.latency_table['pool'] + self.latency_table['fc']
        elastic_time = sum([self.latency_table['blocks'][idx]['%.2f-%d' % (expand_ratios[idx], kernel_sizes[idx])] for idx in indices])
        total_time   = fixed_time + elastic_time
        return total_time

    def measure(self, model : ResNet, in_size=[1, 64, 400], device='cpu'):
        """
        Measure latency of Supernet ResNet via 4-second inputs.

            Note
            ----
                meausure in cpu input 1 sample, but in 1080ti, input 1 batch of 64 samples.
        """
        import sys
        if any('jupyter' in arg for arg in sys.argv):
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        from sugar.metrics import latency

        model.eval()

        in_size = [1] + in_size if device == 'cpu' else [64] + in_size
        # conv
        self.latency_table['conv'] = latency(model.conv, in_size, device=device)
        out_size = model.conv.count_ops(in_size)[1]
        # blocks
        self.latency_table['blocks'] = []
        for idx in tqdm(range(len(model.blocks)), total=len(model.blocks), desc='Measure Latency'):
            latency_unit = self.measure_unit_among_config(model.blocks[idx], out_size, device)
            out_size = model.blocks[idx].count_ops(out_size)[1]
            self.latency_table['blocks'].append(latency_unit)
        out_size = [out_size[0], out_size[1] * out_size[2], out_size[3]]
        # pool
        self.latency_table['pool'] = latency(model.pool, input_size=out_size, device=device)
        out_size = model.pool.count_ops(out_size)[1]
        # fc
        self.latency_table['fc'] = latency(model.fc, input_size=out_size, device=device)

    def measure_unit_among_config(self, unit : nn.Module, in_size, device):
        """Measure latency per unit with each possible pair between expand_ratios and kernel sizes."""
        from sugar.metrics import latency

        latency_unit = {}
        for expand_ratio in self.expand_ratios:
            for kernel_size in self.kernel_sizes:
                # clone the unit
                unit_clone = unit.clone(expand_ratio, kernel_size)
                # measure latency of the cloned unit
                latency_unit['%.2f-%d' % (expand_ratio, kernel_size)] = latency(unit_clone, input_size=in_size, device=device)
        return latency_unit

    def __repr__(self) -> str:
        repr = ''
        if not self.latency_table:
            repr += 'No latency table. Please measure latency first!'
        else:
            for key, value in self.latency_table.items():
                if not isinstance(value, list):
                    repr += '%s: %.2fms\n' % (key, value)
                else:
                    repr += '%s:\n' % key
                    for v in value:
                        for subkey, subvalue in v.items():
                            repr += '\tExpand Ratio and Kernel Size %s: %.2fms\n' % (subkey, subvalue)
        return repr

    def to_yaml(self, yaml_file='/workspace/rwang/sugar/works/nas/exps/conf/tmp.yml'):
        import os
        if not os.path.exists(yaml_file):
            import yaml
            with open(yaml_file, 'w') as f:
                yaml.dump(self.latency_table, f)
        else:
            raise FileExistsError(yaml_file)


class AccuracyPredictor(object):
    """
    Predictor to generate predicted accuracy of the ResNet sub-network with the given config.
        It supports 
            1) predict the accuracy of the given sub-network's config.
            2) train the predictor based all pair of sub-networks's config and accuracy.
            3) load the model from the given diretory when create the predictor.
            4) save the model after training

        Note that the model is achieved via sklearn.neural_network.MLPRegressor(110, 400, 400, 400, 1)
    """

    def __init__(self, pather : ResNetPath, pretrained=''):
        """
        Parameter
        ---------
            pather : ResNetPath
            pretrained : str
                the directory of the trained model
        """
        from sklearn.neural_network import MLPRegressor
        
        self.pather = pather
        self.model = MLPRegressor(hidden_layer_sizes=(400, 400, 400,), activation='relu', max_iter=500, verbose=True)

        if pretrained:
            import os
            import pickle
            if os.path.exists(pretrained):
                with open(pretrained, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                raise FileNotFoundError(pretrained)
        else:
            print('MUST train the prodictor FIRST!')

    def predict(self, inputs):
        """
        Predict via the given sub-network config.

            Parameter
            ---------
                inputs : list or list[list[,]]
                    a series of sub-network config from ResNetPath.

            Return
            ------
                y_pred : numpy.ndarray
                    1-dimensional vector of predicted accuracy of the given resnet's architecture
        """
        inputs = self.pather.config2onehot(inputs)
        y_pred = self.model.predict(inputs)
        return y_pred

    def fit(self, X : list, y : list):
        """
        Train the model that predicts accuracy of the given sub-network's config.
        
            Parameter
            ---------
                X : list
                    a list of sub-network's config.
                y : list
                    a list of accuracy.

            Return
            ------
                score : float
                    a score of test set.
        """
        import numpy
        from sklearn.model_selection import train_test_split
        
        X = numpy.array(self.pather.config2onehot(X))
        y = numpy.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
        self.model.fit(X_train, y_train)

        return self.model.score(X_test, y_test)

    def save(self, model_path):
        """Save the trained model via pickle"""
        import os
        import pickle
        if os.path.exists(model_path):
            raise FileExistsError(model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)


def resnet3m(in_feats=64, out_embeds=256, pather=None):
    """
    Elastic ResNet with around 3,000,000 parameters, exactly 2,814,264, has 3-4-6-3 depths, 16-32-64-128 channels, 5x5 kernel sizes.

        Parameter
        ---------
            in_feats : int
            out_embeds : int
            pather : ResNetPath

        Return
        ------
            modelarch : ResNet
    """
    modelarch = ResNet(depths=[3, 4, 6, 3],
                       widths=[16, 16, 16, 32, 32,  32,  32,  64,
                               64, 64, 64, 64, 64, 128, 128, 128],
                       kernels=[5, 5, 5, 5, 5, 5, 5, 5, 
                                5, 5, 5, 5, 5, 5, 5, 5],
                       pather=pather,
                       in_feats=in_feats, 
                       out_embeds=out_embeds)
    return modelarch


def resnet10m(in_feats=64, out_embeds=256, pather=None):
    """
    Elastic ResNet with around 10,000,000 parameters, exactly 10,188,880, has 3-4-6-3 depths, 32-64-128-256 channels, 5x5 kernel sizes.

        Parameter
        ---------
            in_feats : int
            out_embeds : int
            pather : ResNetPath

        Return
        ------
            modelarch : ResNet
    """
    modelarch = ResNet(depths=[3, 4, 6, 3],
                       widths=[32, 32, 32, 64, 64,  64,  64,  128,
                               128, 128, 128, 128, 128, 256, 256, 256],
                       kernels=[5, 5, 5, 5, 5, 5, 5, 5, 
                                5, 5, 5, 5, 5, 5, 5, 5],
                       pather=pather,
                       in_feats=in_feats, 
                       out_embeds=out_embeds)
    return modelarch


def resnet30m(in_feats=80, out_embeds=256, pather=None):
    """
    Elastic ResNet with around 30,000,000 parameters, exactly 25,019,456, has 3-4-6-3 depths, 16-32-64-128 channels, 5x5 kernel sizes.

        Parameter
        ---------
            in_feats : int
            out_embeds : int
            pather : ResNetPath

        Return
        ------
            modelarch : ResNet
    """
    modelarch = ResNet(depths=[3, 4, 6, 3],
                       widths=[128, 128, 128, 128, 128,  128,  128,  256,
                               256, 256, 256, 256, 256, 256, 256, 256],
                       kernels=[5, 5, 5, 5, 5, 5, 5, 5, 
                                5, 5, 5, 5, 5, 5, 5, 5],
                       pather=pather,
                       in_feats=in_feats, 
                       out_embeds=out_embeds)
    return modelarch


# customized operations for profile
custom_ops = {
    Hsigmoid: thop.vision.basic_hooks.count_relu,
    StatsPool: thop.vision.basic_hooks.count_avgpool,
}


if __name__ == "__main__":
    from sugar.metrics import profile
    from sugar.loss.softmax import Softmax
    from sugar.models import SpeakerModel
    from sugar.transforms import LogMelFbanks

    in_size = [1, 64, 400]
    modelarch = resnet3m(in_feats=64, out_embeds=256, pather=None)
    transform = LogMelFbanks(n_mels=modelarch.in_feats)
    lossfunc  = Softmax(modelarch.out_embeds, 1251)
    model     = SpeakerModel(modelarch, lossfunc, transform)
    model.eval()
    print(model)

    resnet_path = ResNetPath(depths=[3, 4, 6, 3], widths=[0.25, 0.50, 1], kernels=[1, 3, 5], depth_minus=2)
    # print(resnet_path)
    
    print('========== MACs and Parameters Demo ==========')
    macs, params = profile(modelarch, (1, *in_size), custom_ops=custom_ops, verbose=False)
    print('MACs:', macs, 'Params:', params, 'via profile')
    macs = modelarch.count_ops((1, *in_size))
    params = modelarch.count_params()
    macs, params = thop.clever_format([macs, params], "%.2f")
    print('MACs:', macs, 'Params:', params, 'via self-defined count')
    for _ in range(10):
        depth, expand_ratio, kernel_size = resnet_path.depth_width_kernel
        cloned_model = modelarch.clone(depth, expand_ratio, kernel_size)
        macs, params = cloned_model.count_memory(in_size=[1, *in_size], verbose=True)
        print('MACs:', macs, '\tParams:', params, 'with\t%s | %s | %s' % (depth, ' '.join(['%.2f' % er for er in expand_ratio]), kernel_size))
    

    print('========== Latency Query Demo ==========')
    latency_table = ResNetLatency(table='/workspace/rwang/sugar/works/nas/exps/conf/resnet3m_latency_cpu_64_400.yml')
    # print(latency_table)
    
    for _ in range(10):
        depth, expand_ratio, kernel_size = resnet_path.depth_width_kernel
        cost_time = latency_table.query(depth, expand_ratio, kernel_size)
        print('Latency: %.3f ms\twith %s | %s | %s' % (cost_time, depth, ' '.join(['%.2f' % er for er in expand_ratio]), kernel_size))

    quit()
    
    print(__doc__)
    """Github: Once for All"""
    # from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAProxylessNASNets, OFAResNets
    # if net_id == 'ofa_proxyless_d234_e346_k357_w1.3':
    #     net = OFAProxylessNASNets(
    #         dropout_rate=0, width_mult=1.3, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
    #     )
    # elif net_id == 'ofa_mbv3_d234_e346_k357_w1.0':
    #     net = OFAMobileNetV3(
    #         dropout_rate=0, width_mult=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
    #     )
    # elif net_id == 'ofa_mbv3_d234_e346_k357_w1.2':
    #     net = OFAMobileNetV3(
    #         dropout_rate=0, width_mult=1.2, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
    #     )
    # elif net_id == 'ofa_resnet50':
    #     net = OFAResNets(
    #         dropout_rate=0, depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35], width_mult_list=[0.65, 0.8, 1.0]
    #     )
    #     net_id = 'ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0'
    # else:
    #     raise ValueError('Not supported: %s' % net_id)