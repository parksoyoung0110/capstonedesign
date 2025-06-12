from typing import Any, Tuple, Callable, Iterator
from os import path
import csv
import random
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor


def make_reproducible(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def memoize(func: Callable) -> Callable:
    cache = {}

    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper


@memoize
def load_image(file: str) -> torch.FloatTensor:
    return to_tensor(Image.open(file)).unsqueeze(0)


def load_reference() -> Iterator[Tuple[torch.FloatTensor, torch.FloatTensor, float]]:
    assets_root = path.join(path.dirname(__file__), "assets")
    images_root = path.join(assets_root, "images")

    with open(path.join(assets_root, "reference.csv")) as csvfh:
        for row in csv.DictReader(csvfh):
            image1 = load_image(path.join(images_root, row["image1"]))
            image2 = load_image(path.join(images_root, row["image2"]))
            score = float(row["score"])
            yield image1, image2, score
