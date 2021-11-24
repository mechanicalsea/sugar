"""
Interface to Dynamic Neural Architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DynamicLinear', 'DynamicBatchNorm1d', 'DynamicConv1d']

TOPK = 'topk'
INDEX = 'index'

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

class DynamicModule(torch.nn.Module):
    """
    Dynamic Module supports the selection by top-k or by indices, called the mode of 'topk' or 'index'.

        Notes
        -----
            Set the configuration to None, which means reset the module as the initial configuration.
    """

    def forward(self, x):
        raise NotImplementedError
    
    @property
    def _module_configs(self):
        pass

    @property
    def config(self):
        raise NotImplementedError

    @config.setter
    def config(self, config):
        """
        config : int or Tensor
            int is in topk mode, Tensor is in index mode
        """
        raise NotImplementedError

    def clone(self, config=None):
        raise NotImplementedError

    @property
    def device(self):
        pass

    def out_size(self, in_size):
        pass

    def count_params(self):
        """
        Count the maximum/actual number of parameters required gradients.

            Return
            ------
                cnt : int
                    The number of parameters required gradients.
        """
        cnt = 0
        for param in self.parameters():
            if param.requires_grad: cnt += param.numel()
        return cnt

    def count_ops(self, in_size):
        """Count the actual multiply-add operations."""
        pass


class DynamicLinear(DynamicModule):
    """
    Dynamic linear layer supports the selection by top-k or by indices, called the mode of 'topk' or 'index'.
        It supports in_features and out_features.

        Notes
        -----
            Set the configuration to None, which means reset the module as the initial configuration.
    """
    def __init__(self, in_features: int, out_features: int, bias=True):
        super(DynamicLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(self.in_features, self.out_features, bias)

        self.in_mode = TOPK
        self.out_mode = TOPK
        self.active_in_features = self.in_features
        self.active_out_features = self.out_features
        self.active_in_indices = torch.arange(self.in_features, device=self.device).long()
        self.active_out_indices = torch.arange(self.out_features, device=self.device).long()

    def forward(self, x):
        if ((self.in_mode == TOPK and self.active_in_features == self.in_features) or (self.in_mode == INDEX and self.active_in_indices.numel() == self.in_features)) \
            and ((self.out_mode == TOPK and self.active_out_features == self.out_features) or (self.out_mode == INDEX and self.active_out_indices.numel() == self.out_features)):
            return self.linear(x)
        else:
            weights, biases = self.weight_bias
            return F.linear(x, weights, biases)

    @property
    def weight_bias(self):
        biases = None
        is_bias = self.linear.bias is not None

        if self.in_mode == TOPK:
            in_feats = self.active_in_features
            weights = self.linear.weight[:, :in_feats]
        else:
            weights = self.linear.weight.index_select(1, self.active_in_indices)
        
        if self.out_mode == TOPK:
            out_feats = self.active_out_features
            weights = weights[:out_feats]
            if is_bias: biases = self.linear.bias[:out_feats]
        else:
            weights = torch.index_select(weights, 0, self.active_out_indices)
            if is_bias: biases = self.linear.bias.index_select(0, self.active_out_indices)
            
        return weights, biases

    @property
    def config(self):
        """
        [Linear] Config: (in_features, out_features)
        """
        in_feats = self.active_in_features if self.in_mode == TOPK else self.active_in_indices
        out_feats = self.active_out_features if self.out_mode == TOPK else self.active_out_indices
        return (in_feats, out_feats)

    @config.setter
    def config(self, config):
        """
            config : 2-d tuple of int or Tensor
                The configurations of linear layer are of in_features and out_features.
        """
        if config is None:
            config = (None, None)
        in_feats, out_feats = config
        assert in_feats is None or isinstance(in_feats, (int, torch.LongTensor, torch.cuda.LongTensor))
        assert out_feats is None or isinstance(out_feats, (int, torch.LongTensor, torch.cuda.LongTensor))

        if in_feats is None:
            self.active_in_features = self.in_features
            self.active_in_indices = torch.arange(self.in_features, device=self.device).long()
        elif isinstance(in_feats, int):
            self.in_mode = TOPK
            self.active_in_features = in_feats
        else:
            self.in_mode = INDEX
            self.active_in_indices = in_feats
        
        if out_feats is None:
            self.active_out_features = self.out_features
            self.active_out_indices = torch.arange(self.out_features, device=self.device).long()
        elif isinstance(out_feats, int):
            self.out_mode = TOPK
            self.active_out_features = out_feats
        else:
            self.out_mode = INDEX
            self.active_out_indices = out_feats

        self.active_in_indices.to(self.linear.weight.device)
        self.active_out_indices.to(self.linear.weight.device)

    def clone(self, config=None):
        """
        Clone the weight after activating the configuration.

            [DynamicLinear] Config: (in_features, out_features)
        """
        self.config = config
        assert self.in_mode in [TOPK, INDEX] and self.out_mode in [TOPK, INDEX]

        is_bias = self.linear.bias is not None
        in_feats = self.active_in_features if self.in_mode == TOPK else self.active_in_indices.numel()
        out_feats = self.active_out_features if self.out_mode == TOPK else self.active_out_indices.numel()
        m = nn.Linear(in_feats, out_feats, is_bias)
        m = m.to(self.device)

        weights, biases = self.weight_bias

        m.weight.data.copy_(weights)
        if is_bias: m.bias.data.copy_(biases)
        
        return m.train(self.training)

    def out_size(self, in_size):
        out_feats = self.active_out_features if self.out_mode == TOPK else self.active_out_indices.numel()
        out_size = in_size[:1] + [out_feats]
        return out_size

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_features)
        """
        in_feats = in_size[1]

        total_mul = in_feats
        num_elements = _prod(self.out_size(in_size))
        total_ops = 1.0 * total_mul * num_elements
        return total_ops

    def count_params(self):
        in_feats = self.active_in_features if self.in_mode == TOPK else self.active_in_indices.numel()
        out_feats = self.active_out_features if self.out_mode == TOPK else self.active_out_indices.numel()
        total_params = in_feats * out_feats
        if self.linear.bias is not None:
            total_params = total_params + out_feats
        return total_params * 1.0

    @property
    def device(self):
        return self.linear.weight.device


class DynamicBatchNorm1d(DynamicModule):
    """
    Dynamic Batch Normalization 1d supports: in_features.
    """
    def __init__(self, in_features):
        super(DynamicBatchNorm1d, self).__init__()

        self.in_features = in_features

        self.bn = nn.BatchNorm1d(self.in_features)

        self.in_mode = TOPK
        self.active_in_features = self.in_features
        self.active_in_indices = torch.arange(self.in_features, device=self.device).long()

    def forward(self, x):
        if (self.in_mode == TOPK and self.active_in_features == self.in_features) or (self.in_mode == INDEX and self.active_in_indices.numel() == self.in_features):
            return self.bn(x)
        else:
            mean, var, weight, bias, exponential_average_factor = self.mean_var_weight_bias_average
            return F.batch_norm(x,
                                mean if not self.bn.training or self.bn.track_running_stats else None,
                                var if not self.bn.training or self.bn.track_running_stats else None, 
                                weight, bias, self.bn.training or not self.bn.track_running_stats,
                                exponential_average_factor, self.bn.eps)

    @property
    def mean_var_weight_bias_average(self):
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

        if self.in_mode == TOPK:
            in_feats = self.active_in_features
            mean = self.bn.running_mean[:in_feats]
            var = self.bn.running_var[:in_feats]
            weight = self.bn.weight[:in_feats]
            bias = self.bn.bias[:in_feats]
        else:
            mean = torch.index_select(self.bn.running_mean, 0, self.active_in_indices)
            var = torch.index_select(self.bn.running_var, 0, self.active_in_indices)
            weight = torch.index_select(self.bn.weight, 0, self.active_in_indices)
            bias = torch.index_select(self.bn.bias, 0, self.active_in_indices)

        return mean, var, weight, bias, exponential_average_factor

    @property
    def config(self):
        """
        Clone the weight after activating the configuration.

            [DynamicBatchNorm1d] Config: in_features
        """
        in_feats = self.active_in_features if self.in_mode == TOPK else self.active_in_indices
        return in_feats

    @config.setter
    def config(self, config):
        in_feats = config
        assert in_feats is None or isinstance(in_feats, (int, torch.LongTensor, torch.cuda.LongTensor))

        if in_feats is None:
            self.active_in_features = self.in_features
            self.active_in_indices = torch.arange(self.in_features, device=self.device).long()
        elif isinstance(in_feats, int):
            self.in_mode = TOPK
            self.active_in_features = in_feats
        else:
            self.in_mode = INDEX
            self.active_in_indices = in_feats

        self.active_in_indices.to(self.bn.weight.device)

    def clone(self, config=None):
        """
        Clone the weight after activating the configuration.
            [BatchNorm1d] Config: in_features
        """
        self.config = config
        assert self.in_mode in [TOPK, INDEX]

        in_feats = self.active_in_features if self.in_mode == TOPK else self.active_in_indices.numel()
        m = nn.BatchNorm1d(in_feats, eps=self.bn.eps, momentum=self.bn.momentum, affine=self.bn.affine, track_running_stats=self.bn.track_running_stats)
        m = m.to(self.device)

        mean, var, weight, bias, _ = self.mean_var_weight_bias_average

        m.running_mean.data.copy_(mean)
        m.running_var.data.copy_(var)
        m.weight.data.copy_(weight)
        m.bias.data.copy_(bias)
        
        return m.train(self.training)
    
    def out_size(self, in_size):
        return in_size

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)
        """
        nelements = _prod(in_size)
        # subtract, divide, gamma, beta
        total_ops = 2.0 * nelements
        return total_ops

    def count_params(self):
        in_feats = self.active_in_features if self.in_mode == TOPK else self.active_in_indices.numel()
        total_params = in_feats * 2
        return total_params * 1.0

    @property
    def device(self):
        return self.bn.weight.device


class DynamicConv1d(DynamicModule):
    """
    Dyanmic Conv1d supports: in_channels, out_channels, kernel size.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1):
        super(DynamicConv1d, self).__init__()

        assert kernel_size in [1, 3, 5, 7, 9]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        assert not (stride > 1 and dilation > 1), "stride = {} and dilation = {}".format(stride, dilation)
        padding = ((kernel_size - 1) // 2) * dilation # same padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)

        self.active_kernel_size = kernel_size
        if self.kernel_size > 5: # 7 -> 5
            self.k75 = nn.Parameter(torch.eye(5), True)
        else:
            self.register_parameter('k75', None)
        if self.kernel_size > 3: # 5 -> 3
            self.k53 = nn.Parameter(torch.eye(3), True)
        else:
            self.register_parameter('k53', None)
        if self.kernel_size > 1: # 3 -> 1
            self.k31 = nn.Parameter(torch.eye(1), True)
        else:
            self.register_parameter('k31', None)

        self.in_mode = TOPK
        self.out_mode = TOPK
        self.active_in_channels = self.in_channels
        self.active_out_channels = self.out_channels
        self.active_in_indices = torch.arange(self.in_channels, device=self.device).long()
        self.active_out_indices = torch.arange(self.out_channels, device=self.device).long()

    def forward(self, x):
        if self.active_kernel_size == self.kernel_size \
            and ((self.in_mode == TOPK and self.active_in_channels == self.in_channels) or (self.in_mode == INDEX and self.active_in_indices.numel() == self.in_channels)) \
            and ((self.out_mode == TOPK and self.active_out_channels == self.out_channels) or (self.out_mode == INDEX and self.active_out_indices.numel() == self.out_channels)):
            return self.conv(x)
        else:
            weights = self.weight
            padding = ((self.active_kernel_size - 1) // 2) * self.conv.dilation[0]
            return F.conv1d(x, weights, None, self.conv.stride, padding, self.conv.dilation, self.conv.groups)
    
    @property
    def weight(self):
        if self.in_mode == TOPK:
            in_channels = self.active_in_channels
            weights = self.conv.weight[:, :in_channels]
        else:
            weights = self.conv.weight.index_select(1, self.active_in_indices)

        if self.out_mode == TOPK:
            out_channels = self.active_out_channels
            weights = weights[:out_channels]
        else:
            weights = weights.index_select(0, self.active_out_indices)

        if self.kernel_size > 5 and self.active_kernel_size < 7:  # 7 to 5
            weights = torch.matmul(weights[:, :, 1:-1], self.k75)
        if self.kernel_size > 3 and self.active_kernel_size < 5:  # 5 to 3
            weights = torch.matmul(weights[:, :, 1:-1], self.k53)
        if self.kernel_size > 1 and self.active_kernel_size < 3:  # 3 to 1
            weights = torch.matmul(weights[:, :, 1:-1], self.k31)

        return weights

    @property
    def config(self):
        """
        [DynamicConv1d] Config: (in_channels, out_channels, kernel_size)
        """
        in_channels = self.active_in_channels if self.in_mode == TOPK else self.active_in_indices
        out_channels = self.active_out_channels if self.out_mode == TOPK else self.active_out_indices
        kernel_size = self.active_kernel_size
        return (in_channels, out_channels, kernel_size)

    @config.setter
    def config(self, config):
        """
            config : 3-d tuple of int or Tensor
                The configurations of Conv1d is of in_channels, out_channels, and kernel_size.
        """
        if config is None:
            config = (None, None, None)
        in_channels, out_channels, kernel_size = config
        assert in_channels is None or isinstance(in_channels, (int, torch.LongTensor, torch.cuda.LongTensor))
        assert out_channels is None or isinstance(out_channels, (int, torch.LongTensor, torch.cuda.LongTensor))
        assert kernel_size is None or isinstance(kernel_size, int)

        self.active_kernel_size = kernel_size if kernel_size is not None else self.kernel_size

        if in_channels is None:
            self.active_in_channels = self.in_channels
            self.active_in_indices = torch.arange(self.in_channels, device=self.device).long()
        elif isinstance(in_channels, int):
            self.in_mode = TOPK
            self.active_in_channels = in_channels
        else:
            self.in_mode = INDEX
            self.active_in_indices = in_channels
        
        if out_channels is None:
            self.active_out_channels = self.out_channels
            self.active_out_indices = torch.arange(self.out_channels, device=self.device).long()
        elif isinstance(out_channels, int):
            self.out_mode = TOPK
            self.active_out_channels = out_channels
        else:
            self.out_mode = INDEX
            self.active_out_indices = out_channels

        self.active_in_indices.to(self.conv.weight.device)
        self.active_out_indices.to(self.conv.weight.device)

    def clone(self, config=None):
        """
        Clone the weight after activating the configuration.

            [Conv1d] Config: (in_channels, out_channels, kernel_size)
        """
        self.config = config
        assert self.in_mode in [TOPK, INDEX] and self.out_mode in [TOPK, INDEX]

        in_channels = self.active_in_channels if self.in_mode == TOPK else self.active_in_indices.numel()
        out_channels = self.active_out_channels if self.out_mode == TOPK else self.active_out_indices.numel()
        kernel_size = self.active_kernel_size
        padding = ((kernel_size - 1) // 2) * self.conv.dilation[0]
        m = nn.Conv1d(in_channels, out_channels, kernel_size, self.conv.stride, padding, self.conv.dilation, self.conv.groups, bias=False)
        m = m.to(self.device)

        weights = self.weight
        m.weight.data.copy_(weights)
        
        return m.train(self.training)

    def out_size(self, in_size):
        out_channels = self.active_out_channels if self.out_mode == TOPK else self.active_out_indices.numel()
        out_size = in_size[:1] + [out_channels] + in_size[2:]
        return out_size

    def count_ops(self, in_size):
        """
            in_size : (batch_size, in_channels, T)
        """

        kernel_ops = self.active_kernel_size
        bias_ops = 0 # no bias

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        in_channels = in_size[1]
        total_ops = 1.0 * _prod(self.out_size(in_size)) * (in_channels // self.conv.groups * kernel_ops + bias_ops)

        return total_ops

    def count_params(self):
        in_channels = self.active_in_channels if self.in_mode == TOPK else self.active_in_indices.numel()
        out_channels = self.active_out_channels if self.out_mode == TOPK else self.active_out_indices.numel()
        kernel_size = self.active_kernel_size
        total_params = out_channels * in_channels * kernel_size
        return total_params * 1.0

    @property
    def device(self):
        return self.conv.weight.device

    
if __name__ == '__main__':
    m = DynamicConv1d(4, 3, 7)
    m.train()
    config = (2, 2, 1)
    indices = (torch.LongTensor([0, 1]), torch.LongTensor([0, 1]), 1)
    print(m)
    print(m.config, m.training)
    print(m.weight)
    inp = torch.randn(3, 2, 3)
    m.config = config
    print(m.config, m.training)

    print(torch.allclose(m.weight, m.clone(config=config).weight.data))
    print(torch.allclose(m.weight, m.clone(config=indices).weight.data))
    print(torch.allclose(m(inp), m.clone(config=config)(inp)))
    print(torch.allclose(m(inp), m.clone(config=indices)(inp)))

    print(m.count_params())
    