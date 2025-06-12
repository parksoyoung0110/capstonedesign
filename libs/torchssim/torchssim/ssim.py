from collections import namedtuple
import torch
from torch.nn.functional import relu
from torchimagefilter import ImageFilter

__all__ = [
    "SSIMReprenstation",
    "SSIMContext",
    "SimplifiedSSIMContext",
    "calculate_ssim_repr",
    "calculate_luminance",
    "calculate_contrast",
    "calculate_structure",
    "calculate_non_structural",
    "calculate_structural",
    "calculate_ssim",
    "calculate_simplified_ssim",
]

SSIMReprenstation = namedtuple(
    "ssim_reprensentation", ("raw", "mean", "mean_sq", "var")
)
SSIMContext = namedtuple(
    "ssim_context",
    (
        "luminance_eps",
        "contrast_eps",
        "structure_eps",
        "luminance_exp",
        "contrast_exp",
        "structure_exp",
    ),
)
SimplifiedSSIMContext = namedtuple(
    "simplified_ssim_context", ("non_structural_eps", "structural_eps")
)


def _possqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(relu(x))


def calculate_ssim_repr(
    image: torch.FloatTensor, image_filter: ImageFilter
) -> SSIMReprenstation:
    mean = image_filter(image)
    mean_sq = mean ** 2.0
    var = image_filter(image ** 2.0) - mean_sq
    return SSIMReprenstation(image, mean, mean_sq, var)


def calculate_luminance(
    input_mean_sq: torch.FloatTensor,
    target_mean_sq: torch.FloatTensor,
    mean_prod: torch.FloatTensor,
    eps: float,
) -> torch.FloatTensor:
    return (2.0 * mean_prod + eps) / (input_mean_sq + target_mean_sq + eps)


def calculate_contrast(
    input_var: torch.FloatTensor,
    target_var: torch.FloatTensor,
    std_prod: torch.FloatTensor,
    eps: float,
) -> torch.FloatTensor:
    return (2.0 * std_prod + eps) / (input_var + target_var + eps)


def calculate_structure(
    std_prod: torch.FloatTensor, covar: torch.FloatTensor, eps: float
) -> torch.FloatTensor:
    return (covar + eps) / (std_prod + eps)


def calculate_ssim(
    input_repr: SSIMReprenstation,
    target_repr: SSIMReprenstation,
    ctx: SSIMContext,
    image_filter: ImageFilter,
) -> torch.FloatTensor:
    input_mean_sq, target_mean_sq = input_repr.mean_sq, target_repr.mean_sq
    input_var, target_var = input_repr.var, target_repr.var

    mean_prod = input_repr.mean * target_repr.mean
    std_prod = _possqrt(input_var * target_var)
    covar = image_filter(input_repr.raw * target_repr.raw) - mean_prod

    luminance = calculate_luminance(
        input_mean_sq, target_mean_sq, mean_prod, ctx.luminance_eps
    )
    contrast = calculate_contrast(input_var, target_var, std_prod, ctx.contrast_eps)
    structure = calculate_structure(std_prod, covar, ctx.structure_eps)

    return (
        luminance ** ctx.luminance_exp
        * contrast ** ctx.contrast_exp
        * structure ** ctx.structure_exp
    )


def calculate_non_structural(
    input_mean_sq: torch.FloatTensor,
    target_mean_sq: torch.FloatTensor,
    mean_prod: torch.FloatTensor,
    eps: float,
) -> torch.FloatTensor:
    return calculate_luminance(input_mean_sq, target_mean_sq, mean_prod, eps)


def calculate_structural(
    input_var: torch.FloatTensor,
    target_var: torch.FloatTensor,
    covar: torch.FloatTensor,
    eps: float,
) -> torch.FloatTensor:
    return (2.0 * covar + eps) / (input_var + target_var + eps)


def calculate_simplified_ssim(
    input_repr: SSIMReprenstation,
    target_repr: SSIMReprenstation,
    ctx: SimplifiedSSIMContext,
    image_filter: ImageFilter,
) -> torch.FloatTensor:
    input_mean_sq, target_mean_sq = input_repr.mean_sq, target_repr.mean_sq
    input_var, target_var = input_repr.var, target_repr.var

    mean_prod = input_repr.mean * target_repr.mean
    covar = image_filter(input_repr.raw * target_repr.raw) - mean_prod

    non_structural = calculate_non_structural(
        input_mean_sq, target_mean_sq, mean_prod, ctx.non_structural_eps
    )
    structural = calculate_structural(input_var, target_var, covar, ctx.structural_eps)

    return non_structural * structural
