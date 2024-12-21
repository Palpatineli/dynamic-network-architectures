from typing import List, Tuple
from torch import Tensor, nn, norm, zeros_like
from torch.nn.functional import interpolate

from dynamic_network_architectures.building_blocks.grn import GRNwithNHWDC
from dynamic_network_architectures.building_blocks.mednext import LayerNormChannelFirst


def upsample_mask(x_shape: Tuple[int, ...], mask: Tensor) -> List[Tuple[int, int]]:
    upsampled_mask = interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=x_shape[-3:],
                                 mode='nearest').squeeze(0).squeeze(0)
    return upsampled_mask.squeeze(1).nonzero(as_tuple=True)


def simple_masked_forward(func):
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = func(self, x)
        upsampled_mask = interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(x.size(2), x.size(3), x.size(4)),
                                     mode='nearest').squeeze(0).squeeze(0)
        return x * upsampled_mask
    return forward


def norm_masked_forward(func):
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if x.ndim == 5:  # BHWDC or BCHWD
            raise NotImplementedError(f"{self.__class__} supports only 5D tensors")
        ii = upsample_mask(x.shape, mask)
        bhwdc = x.permute(0, 2, 3, 4, 1)
        nc = bhwdc[ii]
        nc = func(func, nc)
        x = zeros_like(bhwdc)
        x[ii] = nc
        return x.permute(0,4,1,2,3)
    return forward


class MaskedModule:
    def forward(self, x: Tensor, mask: Tensor) -> Tensor: ...


class SparseConv3d(nn.Conv3d, MaskedModule):
    @simple_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseMaxPooling(nn.MaxPool3d, MaskedModule):
    @simple_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseAvgPooling(nn.AvgPool3d, MaskedModule):
    @simple_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseGroupNorm(nn.GroupNorm, MaskedModule):
    @norm_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)

    def __repr__(self):
        return super(SparseGroupNorm, self).__repr__()[
               :-1] + f', sp={self.sparse})'


class SparseInstanceNorm(nn.InstanceNorm3d, MaskedModule):
    @simple_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseAdaptiveAvgPooling(nn.AdaptiveAvgPool3d, MaskedModule):
    @simple_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseGRN(GRNwithNHWDC, MaskedModule):
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:  # type: ignore
        if x.ndim != 5:  # Assuming we are dealing with 5D tensors
            raise NotImplementedError("SparseGRN supports only 5D tensors")

        # Get active indices
        ii = upsample_mask(x.shape, mask)

        # Apply GRN on active indices
        nc = x[ii]
        Gx = norm(nc, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        normalized_nc = (self.gamma * (nc * Nx) + self.beta) if self.use_bias else (self.gamma * (nc * Nx))

        # Create a zero tensor and fill in the normalized values
        x_new = zeros_like(x)
        x_new[ii] = normalized_nc
        return x_new

    def __repr__(self):
        return f"SparseGRN(dim={self.gamma.size(1)}, use_bias={self.use_bias})"


class SparseBatchNorm3d(nn.BatchNorm3d, MaskedModule):
    @norm_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseSyncBatchNorm3d(nn.SyncBatchNorm, MaskedModule):
    @norm_masked_forward
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return super().forward(x)


class SparseLayerNorm(nn.LayerNorm, MaskedModule):
    def forward(self, input: Tensor, mask: Tensor) -> Tensor:  # type: ignore
        ii = upsample_mask(input.shape, mask)
        nc = input[ii]
        nc = super().forward(nc)
        x = zeros_like(input)
        x[ii] = nc
        return x


class SparseLayerNormChannelFirst(LayerNormChannelFirst, MaskedModule):
    @norm_masked_forward
    def forward(self, input: Tensor) -> Tensor:  # type: ignore
        return super().forward(input)


conversion_dict = {
    nn.Conv3d: SparseConv3d,
    nn.MaxPool3d: SparseMaxPooling,
    nn.AvgPool3d: SparseAvgPooling,
    nn.GroupNorm: SparseGroupNorm,
    nn.InstanceNorm3d: SparseInstanceNorm,
    nn.AdaptiveAvgPool3d: SparseAdaptiveAvgPooling,
    GRNwithNHWDC: SparseGRN,
    nn.BatchNorm3d: SparseBatchNorm3d,
    nn.SyncBatchNorm: SparseSyncBatchNorm3d,
    nn.LayerNorm: SparseLayerNorm,
    LayerNormChannelFirst: SparseLayerNormChannelFirst,
}


def sparsify(model: nn.Module) -> nn.Module:
    if getattr(model, 'skip_sparse_conversion', False):
        return model
    SparseClass = conversion_dict.get(model.__class__, None)  # type: ignore
    if SparseClass is not None:
        model.__class__ = SparseClass
    else:
        raise NotImplementedError(f'model class {model.__class__} does not have sparse class implemented')
    for _, child in model.named_children():
        sparsify(child)
    return model
