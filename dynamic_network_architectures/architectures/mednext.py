"""Plan-configurable MedNeXt with nnU-Net resolution and supervision conventions."""
from math import prod

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.mednext import (
    MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock, OutBlock, spatial_tuple,
)


def stage_values(value, count, name):
    values = [value] * count if isinstance(value, int) else list(value)
    if len(values) != count or any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError(f'{name} must have {count} positive integer entries')
    return values


def spatial_stages(value, count, ndim, name):
    values = [value] * count if isinstance(value, int) else list(value)
    if len(values) != count:
        raise ValueError(f'{name} must have {count} entries')
    return [spatial_tuple(v, ndim, name) for v in values]


def run_block(block, x, enabled):
    if enabled and block.training and torch.is_grad_enabled():
        return checkpoint(block, x, use_reentrant=False)
    return block(x)


class MedNeXtEncoder(nn.Module):
    def __init__(self, input_channels, features, conv_op, kernels, strides, counts,
                 expansions, do_res, do_res_up_down, norm_type, grn, drop_path_rate,
                 checkpointing):
        super().__init__()
        self.conv_op = conv_op
        self.output_channels = features
        self.strides = strides
        self.checkpointing = checkpointing
        self.stem = conv_op(input_channels, features[0], 1)
        self.stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i, channels in enumerate(features):
            self.stages.append(nn.Sequential(*[
                MedNeXtBlock(conv_op, channels, channels, expansions[i], kernels[i],
                             do_res=do_res, norm_type=norm_type, grn=grn,
                             drop_path_rate=drop_path_rate)
                for _ in range(counts[i])]))
            if i > 0:
                self.down_blocks.append(MedNeXtDownBlock(
                    conv_op, features[i - 1], channels, expansions[i], kernels[i - 1],
                    stride=strides[i], do_res=do_res_up_down, norm_type=norm_type, grn=grn))

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.stages):
            if i:
                x = run_block(self.down_blocks[i - 1], x, self.checkpointing)
            for block in stage:
                x = run_block(block, x, self.checkpointing)
            skips.append(x)
        return skips


class MedNeXtDecoder(nn.Module):
    def __init__(self, features, conv_op, kernels, strides, counts, expansions,
                 num_classes, deep_supervision, do_res, do_res_up_down, norm_type,
                 grn, drop_path_rate, checkpointing):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.checkpointing = checkpointing
        # Decoder lists are ordered by increasing resolution (bottleneck to full resolution).
        self.up_blocks = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for j, i in enumerate(range(len(features) - 2, -1, -1)):
            self.up_blocks.append(MedNeXtUpBlock(
                conv_op, features[i + 1], features[i], expansions[j], kernels[i],
                stride=strides[i + 1], do_res=do_res_up_down, norm_type=norm_type, grn=grn))
            self.stages.append(nn.Sequential(*[
                MedNeXtBlock(conv_op, features[i], features[i], expansions[j], kernels[i],
                             do_res=do_res, norm_type=norm_type, grn=grn,
                             drop_path_rate=drop_path_rate)
                for _ in range(counts[j])]))
            # Always register heads so inference and training state dictionaries agree.
            self.seg_layers.append(OutBlock(conv_op, features[i], num_classes))

    def forward(self, skips):
        x = skips[-1]
        outputs = []
        for j, (up, stage, head) in enumerate(zip(self.up_blocks, self.stages, self.seg_layers)):
            x = run_block(up, x, self.checkpointing)
            skip = skips[-2 - j]
            # Odd input sizes are allowed; planned patches are stride-divisible.
            x = x[(slice(None), slice(None)) + tuple(slice(0, s) for s in skip.shape[2:])]
            x = x + skip
            for block in stage:
                x = run_block(block, x, self.checkpointing)
            if self.deep_supervision or j == len(self.stages) - 1:
                outputs.append(head(x))
        return outputs[::-1] if self.deep_supervision else outputs[-1]


class MedNeXt(AbstractDynamicNetworkArchitectures):
    """n_stages includes the bottleneck; strides[0] must be identity.

    Encoder counts/expansions are ordered full resolution to bottleneck;
    decoder counts/expansions are ordered bottleneck to full resolution.
    """
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op,
                 kernel_sizes, strides, num_classes, n_conv_per_stage=2,
                 n_conv_per_stage_decoder=2, expansion_ratio=4,
                 expansion_ratio_decoder=None, deep_supervision=False,
                 do_res=True, do_res_up_down=True, norm_type='group', grn=True,
                 drop_path_rate=0., outside_block_checkpointing=False):
        super().__init__()
        if not isinstance(n_stages, int) or n_stages < 2:
            raise ValueError('MedNeXt needs at least two resolution stages')
        if input_channels < 1 or num_classes < 1:
            raise ValueError('input_channels and num_classes must be positive')
        ndim = convert_conv_op_to_dim(conv_op)
        if ndim != 3:
            raise ValueError('This MedNeXt architecture supports 3D only')
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * 2 ** i for i in range(n_stages)]
        features = stage_values(features_per_stage, n_stages, 'features_per_stage')
        kernels = spatial_stages(kernel_sizes, n_stages, ndim, 'kernel_sizes')
        strides = spatial_stages(strides, n_stages, ndim, 'strides')
        if strides[0] != (1,) * ndim or any(s not in (1, 2) for stage in strides for s in stage):
            raise ValueError('strides[0] must be identity; subsequent strides must use 1 or 2')
        if any(stage == (1,) * ndim for stage in strides[1:]):
            raise ValueError('Each subsequent stage must downsample at least one axis')
        counts = stage_values(n_conv_per_stage, n_stages, 'n_conv_per_stage')
        dec_counts = stage_values(n_conv_per_stage_decoder, n_stages - 1, 'n_conv_per_stage_decoder')
        expansions = stage_values(expansion_ratio, n_stages, 'expansion_ratio')
        dec_expansions = (expansions[:-1][::-1] if expansion_ratio_decoder is None else
                          stage_values(expansion_ratio_decoder, n_stages - 1, 'expansion_ratio_decoder'))
        self.n_stages = n_stages
        self.num_classes = num_classes
        self.encoder = MedNeXtEncoder(
            input_channels, features, conv_op, kernels, strides, counts, expansions,
            do_res, do_res_up_down, norm_type, grn, drop_path_rate, outside_block_checkpointing)
        self.decoder = MedNeXtDecoder(
            features, conv_op, kernels, strides, dec_counts, dec_expansions,
            num_classes, deep_supervision, do_res, do_res_up_down, norm_type, grn,
            drop_path_rate, outside_block_checkpointing)
        self.key_to_encoder = 'encoder.stages'
        self.key_to_stem = 'encoder.stem'
        self.keys_to_in_proj = ('encoder.stem',)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def compute_conv_feature_map_size(self, input_size):
        """Convolution output elements per sample, including expansion and skip projections.

        This is a topology metric, not a measured training memory requirement.
        Decoder heads are included regardless of the inference supervision toggle.
        """
        if len(input_size) != 3 or any(s < 1 for s in input_size):
            raise ValueError('input_size must be three positive spatial dimensions')
        size = tuple(int(s) for s in input_size)
        sizes = []
        total = prod(size) * self.encoder.output_channels[0]
        for i, stage in enumerate(self.encoder.stages):
            if i:
                down = self.encoder.down_blocks[i - 1]
                size = tuple((s + st - 1) // st for s, st in zip(size, down.stride))
                total += down.compute_conv_feature_map_size(size)
            sizes.append(size)
            total += sum(block.compute_conv_feature_map_size(size) for block in stage)
        for j, (up, stage) in enumerate(zip(self.decoder.up_blocks, self.decoder.stages)):
            up_size = tuple(s * st for s, st in zip(size, up.stride))
            total += up.compute_conv_feature_map_size(up_size)
            size = sizes[-2 - j]
            total += sum(block.compute_conv_feature_map_size(size) for block in stage)
            total += prod(size) * self.num_classes
        return int(total)
