"""MedNeXt blocks with channel-first GRN and anisotropic resampling."""
from math import prod

import torch
from torch import nn

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_convtransp


def spatial_tuple(value, ndim, name):
    values = (value,) * ndim if isinstance(value, int) else tuple(value)
    if len(values) != ndim or any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError(f"{name} must contain {ndim} positive integers, got {value}")
    return values


class LayerNormChannelFirst(nn.LayerNorm):
    def forward(self, x):
        # Native layer_norm accumulates safely under autocast.
        return torch.nn.functional.layer_norm(
            x.movedim(1, -1), self.normalized_shape, self.weight, self.bias, self.eps
        ).movedim(-1, 1)


class GlobalResponseNorm(nn.Module):
    def __init__(self, channels, ndim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, *([1] * ndim)))
        self.beta = nn.Parameter(torch.zeros(1, channels, *([1] * ndim)))

    def forward(self, x):
        # FP32 reduction prevents overflow for large FP16 feature maps.
        gx = torch.linalg.vector_norm(x.float(), dim=tuple(range(2, x.ndim)), keepdim=True)
        nx = (gx / (gx.mean(dim=1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * (x * nx) + self.beta.to(x.dtype)


class DropPath(nn.Module):
    def __init__(self, probability=0.):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError('drop_path_rate must be in [0, 1)')
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class MedNeXtBlock(nn.Module):
    def __init__(self, conv_op, input_channels, out_channels, expansion_ratio=4,
                 kernel_size=3, do_res=True, norm_type='group', n_groups=None,
                 grn=False, drop_path_rate=0.):
        super().__init__()
        ndim = convert_conv_op_to_dim(conv_op)
        kernel_size = spatial_tuple(kernel_size, ndim, 'kernel_size')
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError('MedNeXt requires odd kernel sizes')
        if not isinstance(expansion_ratio, int) or expansion_ratio < 1:
            raise ValueError('expansion_ratio must be a positive integer')
        if do_res and input_channels != out_channels:
            raise ValueError('Identity residual requires equal input and output channels')
        self.input_channels, self.out_channels = input_channels, out_channels
        self.expanded_channels = input_channels * expansion_ratio
        self.do_res = do_res
        self.kernel_size = kernel_size
        self.stride = (1,) * ndim
        self.conv1 = conv_op(input_channels, input_channels, kernel_size,
                             padding=tuple(k // 2 for k in kernel_size),
                             groups=input_channels if n_groups is None else n_groups)
        if norm_type == 'group':
            self.norm = nn.GroupNorm(input_channels, input_channels)
        elif norm_type == 'layer':
            self.norm = LayerNormChannelFirst(input_channels, eps=1e-6)
        else:
            raise ValueError(f'Unknown norm_type: {norm_type}')
        self.conv2 = conv_op(input_channels, self.expanded_channels, 1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(self.expanded_channels, ndim) if grn else nn.Identity()
        self.conv3 = conv_op(self.expanded_channels, out_channels, 1)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        y = self.conv3(self.grn(self.act(self.conv2(self.norm(self.conv1(x))))))
        return x + self.drop_path(y) if self.do_res else y

    def compute_conv_feature_map_size(self, output_size):
        return prod(output_size) * (self.input_channels + self.expanded_channels + self.out_channels)


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(self, conv_op, input_channels, out_channels, expansion_ratio=4,
                 kernel_size=3, stride=2, do_res=True, norm_type='group', grn=False):
        super().__init__(conv_op, input_channels, out_channels, expansion_ratio,
                         kernel_size, do_res=False, norm_type=norm_type, grn=grn)
        self.stride = spatial_tuple(stride, convert_conv_op_to_dim(conv_op), 'stride')
        self.conv1 = conv_op(input_channels, input_channels, self.kernel_size,
                             stride=self.stride, padding=tuple(k // 2 for k in self.kernel_size),
                             groups=input_channels)
        self.res_conv = conv_op(input_channels, out_channels, 1, stride=self.stride) if do_res else None

    def forward(self, x):
        y = super().forward(x)
        return y + self.res_conv(x) if self.res_conv is not None else y

    def compute_conv_feature_map_size(self, output_size):
        return super().compute_conv_feature_map_size(output_size) + (
            prod(output_size) * self.out_channels if self.res_conv is not None else 0)


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, conv_op, input_channels, out_channels, expansion_ratio=4,
                 kernel_size=3, stride=2, do_res=True, norm_type='group', grn=False):
        super().__init__(conv_op, input_channels, out_channels, expansion_ratio,
                         kernel_size, do_res=False, norm_type=norm_type, grn=grn)
        self.stride = spatial_tuple(stride, convert_conv_op_to_dim(conv_op), 'stride')
        trans_op = get_matching_convtransp(conv_op)
        output_padding = tuple(s - 1 for s in self.stride)
        self.conv1 = trans_op(input_channels, input_channels, self.kernel_size,
                              stride=self.stride, padding=tuple(k // 2 for k in self.kernel_size),
                              output_padding=output_padding, groups=input_channels)
        self.res_conv = trans_op(input_channels, out_channels, 1, stride=self.stride,
                                 output_padding=output_padding) if do_res else None

    def forward(self, x):
        y = super().forward(x)
        return y + self.res_conv(x) if self.res_conv is not None else y

    def compute_conv_feature_map_size(self, output_size):
        return super().compute_conv_feature_map_size(output_size) + (
            prod(output_size) * self.out_channels if self.res_conv is not None else 0)


class OutBlock(nn.Module):
    def __init__(self, conv_op, in_channels, n_classes):
        super().__init__()
        self.conv_out = conv_op(in_channels, n_classes, 1)

    def forward(self, x):
        return self.conv_out(x)
