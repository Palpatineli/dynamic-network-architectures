from typing import List, Literal, Sequence, overload
from torch import Tensor, nn, cat
from dynamic_network_architectures.building_blocks.residual import BasicResBlock
from dynamic_network_architectures.building_blocks.stu_net import UpsampleLayer

class STUEncoder(nn.Module):
    def __init__(self,
                 input_channel: int,
                 num_classes: int,
                 pool_op_kernel_sizes: Sequence[int],
                 conv_kernel_sizes: Sequence[int],
                 depth: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 channels: Sequence[int] = (32, 64, 128, 256, 512, 512),
                 enable_deep_supervision: bool = True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channel
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.channels = channels

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        self.conv_pad_sizes.extend([i // 2 for i in conv_kernel_sizes])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(channels) - 1

        channels = [input_channel] + list(channels)
        strides = [1] + self.pool_kernel_sizes
        # encoder
        self.conv_blocks_context = nn.ModuleList(
            [nn.Sequential(
                BasicResBlock(channels[d], channels[d + 1], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                              stride=strides[d], use_1x1conv=True),
                *[BasicResBlock(channels[d + 1], channels[d + 1], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                    for _ in range(depth[d] - 1)]
            ) for d in range(num_pool + 1)]
        )

    def get_downsample_ratio(self) -> int:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: the TOTAL downsample ratio of the ConvNet.
        E.g., for a ResNet-50, this should return 32.
        """
        return 16

    def get_feature_map_channels(self) -> Sequence[int]:
        """
        This func would ONLY be used in `SparseEncoder's __init__` (see `pretrain/encoder.py`).

        :return: a list of the number of channels of each feature map.
        E.g., for a ResNet-50, this should return [256, 512, 1024, 2048].
        """
        return self.channels[:5]

    @overload
    def forward(self, x: Tensor, hierarchical: Literal[True]) -> List[Tensor]: ...

    @overload
    def forward(self, x: Tensor, hierarchical: Literal[False]) -> Tensor: ...

    def forward(self, x: Tensor, hierarchical: bool = False) -> Tensor | List[Tensor]:
        skips = []

        for d in range(len(self.conv_blocks_context)):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
        if hierarchical:
            return skips
        else:
            return x


class STUNet(nn.Module):
    def __init__(self,
                 input_channel: int,
                 num_classes: int,
                 pool_op_kernel_sizes: Sequence[int],
                 conv_kernel_sizes: Sequence[int],
                 depth: Sequence[int] = (1, 1, 1, 1, 1, 1),
                 channels: Sequence[int] = (32, 64, 128, 256, 512, 512),
                 enable_deep_supervision: bool = True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channel
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        self.conv_pad_sizes.extend([i // 2 for i in conv_kernel_sizes])

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(channels) - 1

        self.encoder = STUEncoder(input_channel, num_classes, pool_op_kernel_sizes[:-1], conv_kernel_sizes, depth,
                                  channels[:-1], enable_deep_supervision)

        self.bottleneck = nn.Sequential(
            BasicResBlock(channels[num_pool], channels[num_pool], self.conv_kernel_sizes[num_pool],
                          self.conv_pad_sizes[num_pool], stride=self.pool_op_kernel_sizes[num_pool - 1],
                          use_1x1conv=True),
            *[BasicResBlock(channels[num_pool], channels[num_pool], self.conv_kernel_sizes[num_pool],
                            self.conv_pad_sizes[num_pool]) for _ in range(depth[num_pool] - 1)])

        # upsample_layers
        self.upsample_layers = nn.ModuleList([
            UpsampleLayer(channels[-1 - u], channels[-2 - u], pool_op_kernel_sizes[-1 - u]) for u in range(num_pool)])

        # decoder
        self.conv_blocks_localization = nn.ModuleList([
            nn.Sequential(BasicResBlock(channels[-2 - u] * 2, channels[-2 - u], self.conv_kernel_sizes[-2 - u],
                                        self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                          *[BasicResBlock(channels[-2 - u], channels[-2 - u], self.conv_kernel_sizes[-2 - u],
                                          self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            for u in range(num_pool)
        ])

        # outputs
        self.seg_outputs = nn.ModuleList([nn.Conv3d(channels[-2 - ds], num_classes, kernel_size=1)
            for ds in range(len(self.conv_blocks_localization))])
        self.upscale_logits_ops = [lambda x: x for _ in range(num_pool - 1)]

    def forward(self, x):
        skips = self.encoder.forward(x, True)
        x = self.bottleneck(x)

        seg_outputs = []
        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
