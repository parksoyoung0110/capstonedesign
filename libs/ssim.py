import torch
from torchssim import (
    get_default_ssim_image_filter,
    calculate_ssim_repr as _calculate_ssim_repr,
    calculate_non_structural,
    calculate_structural,
    calculate_simplified_ssim,
)
import pystiche
from pystiche.image.transforms.functional import rgb_to_grayscale
from PIL import Image
from torchvision import transforms

SSIMReprenstation = pystiche.namedtuple(
    "ssim_reprensentation", ("raw", "mean", "mean_sq", "var")
)

SimplifiedSSIMContext = pystiche.namedtuple(
    "simplified_ssim_context", ("non_structural_eps", "structural_eps")
)


def calculate_ssim_repr(image, image_filter):
    ssim_repr = _calculate_ssim_repr(image, image_filter)
    return SSIMReprenstation(*ssim_repr)


def calculate_dynamic_range(x):
    dim = 2
    x = torch.abs(torch.flatten(x, dim))
    dynamic_range = torch.max(x, dim).values
    return dynamic_range.view(*dynamic_range.size(), 1, 1)


def calculate_ssim_eps(const, dynamic_range, eps=1e-8):
    return torch.clamp((const * dynamic_range) ** 2.0, eps)


def calculate_simplified_ssim_ctx(x, non_structural_const=1e-2, structural_const=3e-2):
    dynamic_range = calculate_dynamic_range(x)
    non_structural_eps = calculate_ssim_eps(non_structural_const, dynamic_range)
    structural_eps = calculate_ssim_eps(structural_const, dynamic_range)
    return SimplifiedSSIMContext(non_structural_eps, structural_eps)

component_weight_ratio=1.0
non_structural_weight =  1.0 / (1.0 + component_weight_ratio)
structural_weight = 1.0 - non_structural_weight
image_filter = get_default_ssim_image_filter()

def calculate_score(input_repr, target_repr, ctx):
    input_mean_sq, target_mean_sq = input_repr.mean_sq, target_repr.mean_sq
    input_var, target_var = input_repr.var, target_repr.var
    mean_prod = input_repr.mean * target_repr.mean
    covar = image_filter(input_repr.raw * target_repr.raw) - mean_prod

    non_structural = calculate_non_structural(
        input_mean_sq, target_mean_sq, mean_prod, ctx.non_structural_eps
    )
    structural = calculate_structural(
        input_var, target_var, covar, ctx.structural_eps
    )

    non_structural_score = non_structural_weight * torch.mean(
        1.0 - non_structural
    )
    structural_score = structural_weight * torch.mean(1.0 - structural)
    return non_structural_score + structural_score




