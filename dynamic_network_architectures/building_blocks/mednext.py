from typing import Literal, Sequence, Type
import torch
from torch import Tensor, nn
from torch.nn.functional import layer_norm
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim, get_matching_convtransp


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        conv_op: Type[nn.Conv2d] | Type[nn.Conv3d],
        input_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: Sequence[int] = (3, 3, 3),
        do_res: int = True,
        norm_type: Literal["group", "layer"] = "group",
        n_groups: int | None = None,
        grn: bool = False,  # global response normalization
    ):

        super().__init__()

        self.do_res = do_res
        self.conv_op = conv_op

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv_op(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=1,
            padding=tuple(x // 2 for x in kernel_size),  # type: ignore
            groups=input_channels if n_groups is None else n_groups,
            device='cuda',
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=input_channels, num_channels=input_channels)
        elif norm_type == "layer":
            self.norm = LayerNorm(
                normalized_shape=input_channels, data_format="channels_first"
            )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv_op(
            in_channels=input_channels,
            out_channels=expansion_ratio * input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            device='cuda',
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=expansion_ratio * input_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            device='cuda',
        )

        self.grn = grn
        ndim = convert_conv_op_to_dim(self.conv_op)
        dimensions = ndim * [1]
        if grn:
            self.grn_beta = nn.Parameter(
                torch.zeros(1, expansion_ratio * input_channels, *dimensions),
                requires_grad=True,
            )
            self.grn_gamma = nn.Parameter(
                torch.zeros(1, expansion_ratio * input_channels, *dimensions),
                requires_grad=True,
            )
            self.grn_dimension = tuple(list(range(-ndim, 0)))

    def forward(self, x, dummy_tensor=None):
        del dummy_tensor
        x1 = x
        x1: Tensor = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            gx = torch.norm(x1, p=2, dim=self.grn_dimension, keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(
        self,
        conv_op: Type[nn.Conv2d] | Type[nn.Conv3d],
        input_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: Sequence[int] = (3, 3, 3),
        stride: int = 2,
        do_res: bool = False,
        norm_type: Literal['group', 'layer'] = "group",
        grn: bool = False,
    ):

        super().__init__(
            conv_op,
            input_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            grn=grn,
        )

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv_op(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                device='cuda',
            )

        self.conv1 = conv_op(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,
            padding=tuple(x // 2 for x in kernel_size),  # type: ignore
            groups=input_channels,
            device='cuda',
        )

    def forward(self, x, dummy_tensor=None):
        del dummy_tensor
        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(
        self,
        conv_op: Type[nn.Conv2d] | Type[nn.Conv3d],
        input_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        kernel_size: Sequence[int] = (5, 5, 5),
        stride: int = 2,
        do_res: int = False,
        norm_type: Literal['group', 'layer'] = "group",
        grn: bool = False,
    ):
        super().__init__(
            conv_op,
            input_channels,
            out_channels,
            expansion_ratio,
            kernel_size,
            do_res,
            norm_type,
            grn=grn,
        )

        self.resample_do_res = do_res
        trans_op: Type[nn.ConvTranspose2d] | Type[nn.ConvTranspose3d] = get_matching_convtransp(conv_op)  # type: ignore

        if do_res:
            self.res_conv = trans_op(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                device='cuda',
            )

        self.conv1 = trans_op(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,
            padding=tuple(x // 2 for x in kernel_size),  # type: ignore
            groups=input_channels,
            device='cuda',
        )

    def forward(self, x, dummy_tensor=None):
        del dummy_tensor
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        padding = tuple([1, 0] * len(self.conv1.kernel_size))
        x1 = torch.nn.functional.pad(x1, padding)
        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, padding)
            x1 = x1 + res
        return x1


class OutBlock(nn.Module):
    def __init__(self, conv_op: Type[nn.Conv2d] | Type[nn.Conv3d], in_channels: int, n_classes: int):
        super().__init__()
        self.conv_out = get_matching_convtransp(conv_op)(in_channels, n_classes, kernel_size=1,
                                                         device='cuda')  # type: ignore

    def forward(self, x: Tensor, dummy_tensor=None):
        del dummy_tensor
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        data_format: Literal["channels_first", "channels_last"] = "channels_last",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, device='cuda'))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape, device='cuda'))  # gamma
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = [normalized_shape] if isinstance(normalized_shape, int) else normalized_shape

    def forward(self, x: Tensor, dummy_tensor=False):
        del dummy_tensor
        if self.data_format == "channels_last":
            return layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        else:
            raise NotImplementedError
