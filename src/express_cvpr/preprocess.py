from __future__ import annotations

from typing import Iterable

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]:
    ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(ratios, key=lambda x: x[0] * x[1])


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Iterable[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> tuple[list[Image.Image], tuple[int, int]]:
    width, height = image.size
    aspect_ratio = width / height
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        _target_ratios(min_num, max_num),
        width,
        height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images, target_aspect_ratio


def image_to_internvl_tensor(
    image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, tuple[int, int]]:
    transform = build_transform(input_size=input_size)
    tiles, target_aspect_ratio = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values.to(device=device, dtype=dtype), target_aspect_ratio
