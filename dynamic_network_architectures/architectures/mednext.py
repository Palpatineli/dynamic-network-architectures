from typing import Callable, List, Literal, Sequence, Type
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint  # type: ignore
from dynamic_network_architectures.building_blocks.mednext import MedNeXtBlock, MedNeXtUpBlock, MedNeXtDownBlock
from dynamic_network_architectures.building_blocks.mednext import OutBlock 
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

checkpoint: Callable[[Callable, torch.Tensor, nn.Parameter], torch.Tensor] = checkpoint


class MedNeXt(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        channels: int | Sequence[int],
        conv_op: Type[nn.Conv2d] | Type[nn.Conv3d],
        kernel_sizes: int | Sequence[Sequence[int]],
        strides: int | Sequence[Sequence[int]],
        num_classes: int,
        n_conv_per_stage: int | Sequence[int],  # Can be used to test staging ratio:
        expansion_ratio: int | Sequence[int],  # Expansion ratio as in Swin Transformers
        deep_supervision: bool,  # Can be used to test deep supervision
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        outside_block_checkpointing: bool = False,  # Either inside block or outside block
        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type: Literal['group', 'layer'] = "group",
        grn: bool = False,
    ):

        super().__init__()

        self.n_stages = n_stages
        self.deep_supervision = deep_supervision
        self.outside_block_checkpointing = outside_block_checkpointing

        if isinstance(expansion_ratio, int):
            expansion_ratio = [expansion_ratio] * (2 * n_stages + 1)


        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages

        ndim = convert_conv_op_to_dim(conv_op)

        if isinstance(strides, int):
            strides = [[strides] * ndim] * n_stages

        if isinstance(kernel_sizes, int):
            kernel_sizes = [[kernel_sizes] * ndim] * n_stages

        if isinstance(channels, int):
            scale = np.round(np.exp(np.log(strides).mean(axis=1))).astype(int)
            channels = list(scale.cumprod() * channels)

        self.stem = conv_op(input_channels, channels[0], kernel_size=1)
        self.enc_blocks = []
        self.dec_blocks = []
        self.out_blocks: List[nn.Module] = []
        self.down_blocks = []
        self.up_blocks = []

        for idx, (channel, stride, kernel_size) in enumerate(zip(channels, strides[1:], kernel_sizes)):
            self.enc_blocks.append(nn.Sequential(
                *[
                    MedNeXtBlock(
                        conv_op=conv_op,
                        input_channels=channel,
                        out_channels=channel,
                        expansion_ratio=expansion_ratio[idx],
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        grn=grn,
                    )
                    for _ in range(n_conv_per_stage[idx])
                ]
            ))
            self.down_blocks.append(MedNeXtDownBlock(
                conv_op=conv_op,
                input_channels=channel,
                out_channels=stride[0] * channel,
                expansion_ratio=expansion_ratio[idx + 1],
                kernel_size=kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
            ))
            up_stage_idx = 2 * n_stages - idx
            self.up_blocks.append(MedNeXtUpBlock(
                conv_op,
                input_channels=stride[0] * channel,
                out_channels=channel,
                expansion_ratio=expansion_ratio[up_stage_idx],
                kernel_size=kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                grn=grn,
            ))
            self.dec_blocks.append(nn.Sequential(
                *[
                    MedNeXtBlock(
                        conv_op,
                        input_channels=channel,
                        out_channels=channel,
                        expansion_ratio=expansion_ratio[up_stage_idx],
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        grn=grn,
                    )
                    for _ in range(n_conv_per_stage[up_stage_idx])
                ]
            ))
        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    conv_op=conv_op,
                    input_channels=channels[-1],
                    out_channels=channels[-1],
                    expansion_ratio=expansion_ratio[n_stages],
                    kernel_size=kernel_sizes[-1],
                    do_res=do_res,
                    norm_type=norm_type,
                    grn=grn,
                )
                for _ in range(n_conv_per_stage[n_stages])
            ]
        )


        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        if deep_supervision:
            self.out_blocks = [*(OutBlock(conv_op, channel, num_classes) for channel in channels),
                               OutBlock(conv_op, in_channels=channels[-1] * strides[-1][0], n_classes=num_classes)]
        else:
            self.out_blocks = [OutBlock(conv_op, channels[0], num_classes)]

        self.block_counts = n_conv_per_stage

    def iterative_checkpoint(self, sequential_block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for idx in sequential_block:
            x = checkpoint(idx, x, self.dummy_tensor)
        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor] | torch.Tensor:  # type: ignore
        x = self.stem(x)
        if self.outside_block_checkpointing:
            encoding_result = []
            for idx in range(self.n_stages):
                encoding_result.append(self.iterative_checkpoint(self.enc_blocks[idx], x))
                x = checkpoint(self.down_blocks[idx], encoding_result[idx], self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.deep_supervision:
                out = [checkpoint(self.out_blocks[self.n_stages], x, self.dummy_tensor)]
                for idx in range(self.n_stages - 1, -1, -1):
                    x = checkpoint(self.up_blocks[idx], x, self.dummy_tensor)
                    x = encoding_result[idx] + x
                    x = self.iterative_checkpoint(self.dec_blocks[idx], x)
                    out.append(checkpoint(self.out_blocks[idx], x, self.dummy_tensor))
                return list(reversed(out))
            else:
                for idx in range(self.n_stages - 1, -1, -1):
                    x = checkpoint(self.up_blocks[idx], x, self.dummy_tensor)
                    x = encoding_result[idx] + x
                    x = self.iterative_checkpoint(self.dec_blocks[idx], x)
                return checkpoint(self.out_blocks[0], x, self.dummy_tensor)
        else:
            encoding_result = []
            for idx in range(self.n_stages):
                x = self.enc_blocks[idx](x)
                encoding_result.append(x)
                x = self.down_blocks[idx](x)

            x = self.bottleneck(x)
            if self.deep_supervision:
                out: List[Tensor] = [self.out_blocks[self.n_stages](x)]
                for idx in range(self.n_stages - 1, -1, -1):
                    x = self.up_blocks[idx](x)
                    x = encoding_result[idx] + x
                    encoding_result[idx] = None
                    torch.cuda.empty_cache()
                    x = self.dec_blocks[idx](x)
                    out.append(self.out_blocks[idx](x))
                return list(reversed(out))
            else:
                for idx in range(self.n_stages - 1, -1, -1):
                    x = self.up_blocks[idx](x)
                    x = encoding_result[idx] + x
                    encoding_result[idx] = None
                    torch.cuda.empty_cache()
                    x = self.dec_blocks[idx](x)
                return self.out_blocks[0](x)
