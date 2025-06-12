from abc import abstractmethod
from typing import Union, Optional
import torch
from torch import nn
from torchimagefilter import ImageFilter, GaussFilter
from .ssim import *

__all__ = [
    "get_default_ssim_image_filter",
    "SSIM",
    "SimplifiedSSIM",
    "MSSIM",
    "SimplifiedMSSIM",
]


def get_default_ssim_image_filter() -> GaussFilter:
    return GaussFilter(std=1.5, radius=5, output_shape="same", padding_mode="replicate")


class _SSIMModule(nn.Module):
    def __init__(self, image_filter: Optional[ImageFilter] = None) -> None:
        super().__init__()

        if image_filter is None:
            image_filter = get_default_ssim_image_filter()
        self.image_filter = image_filter

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor):
        input_repr = calculate_ssim_repr(input, self.image_filter)
        target_repr = calculate_ssim_repr(target, self.image_filter)
        return self.calculate_score(
            input_repr, target_repr, self.ctx, self.image_filter
        )

    @property
    @abstractmethod
    def ctx(self) -> Union[SSIMContext, SimplifiedSSIMContext]:
        pass

    @abstractmethod
    def calculate_score(
        self,
        input_repr: SSIMReprenstation,
        target_repr: SSIMReprenstation,
        ctx: Union[SSIMContext, SimplifiedSSIMContext],
        image_filter: ImageFilter,
    ):
        pass


class SSIM(_SSIMModule):
    def __init__(
        self,
        luminance_eps: float = 1e-4,
        contrast_eps: float = 9e-4,
        structure_eps: float = 4.5e-4,
        luminance_exp: float = 1.0,
        contrast_exp: float = 1.0,
        structure_exp: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._ssim_ctx = SSIMContext(
            luminance_eps,
            contrast_eps,
            structure_eps,
            luminance_exp,
            contrast_exp,
            structure_exp,
        )

    @property
    def ctx(self) -> SSIMContext:
        return self._ssim_ctx

    def calculate_score(
        self,
        input_repr: SSIMReprenstation,
        target_repr: SSIMReprenstation,
        ctx: SSIMContext,
        image_filter: ImageFilter,
    ) -> torch.FloatTensor:
        return calculate_ssim(input_repr, target_repr, ctx, image_filter)


class SimplifiedSSIM(_SSIMModule):
    def __init__(self, non_structural_eps=1e-4, structural_eps=9e-4, **kwargs):
        super().__init__(**kwargs)
        self._simplified_ssim_ctx = SimplifiedSSIMContext(
            non_structural_eps, structural_eps
        )

    @property
    def ctx(self) -> SimplifiedSSIMContext:
        return self._simplified_ssim_ctx

    def calculate_score(
        self,
        input_repr: SSIMReprenstation,
        target_repr: SSIMReprenstation,
        ctx: SimplifiedSSIMContext,
        image_filter: ImageFilter,
    ):
        return calculate_simplified_ssim(input_repr, target_repr, ctx, image_filter)


class MSSIM(SSIM):
    def forward(self, *args, **kwargs) -> torch.FloatTensor:
        ssim = super().forward(*args, **kwargs)
        return torch.mean(ssim, dim=(1, 2, 3))


class SimplifiedMSSIM(SimplifiedSSIM):
    def forward(self, *args, **kwargs) -> torch.FloatTensor:
        ssim = super().forward(*args, **kwargs)
        return torch.mean(ssim, dim=(1, 2, 3))
