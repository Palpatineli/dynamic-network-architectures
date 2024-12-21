from typing import Tuple
from torch import nn, Tensor

class UpsampleLayer(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, pool_op_kernel_size: Tuple[int, ...] | int,
                 mode: str = 'nearest') -> None:
        super().__init__()
        self.conv = nn.Conv3d(input_channel, output_channel, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x
