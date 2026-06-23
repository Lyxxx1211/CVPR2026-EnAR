from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoTokenizer

from internvl.model.internvl_chat import InternVLChatModel

from .preprocess import image_to_internvl_tensor


@dataclass
class InternVLBundle:
    model: InternVLChatModel
    tokenizer: AutoTokenizer


def load_internvl(
    model_path: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attn: bool = False,
) -> InternVLBundle:
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return InternVLBundle(model=model, tokenizer=tokenizer)


@torch.no_grad()
def answer_with_reconstruction(
    bundle: InternVLBundle,
    image: Image.Image,
    reconstructed_image: Image.Image,
    diffusion_indices: torch.LongTensor,
    question: str,
    image_size: int = 448,
    max_tiles: int = 12,
    max_new_tokens: int = 1024,
    verbose: bool = False,
) -> str:
    device = next(bundle.model.parameters()).device
    dtype = next(bundle.model.parameters()).dtype
    pixel_values, _ = image_to_internvl_tensor(
        image,
        input_size=image_size,
        max_num=max_tiles,
        dtype=dtype,
        device=device,
    )
    pixel_values_recons, _ = image_to_internvl_tensor(
        reconstructed_image,
        input_size=image_size,
        max_num=max_tiles,
        dtype=dtype,
        device=device,
    )
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "verbose": verbose,
    }
    return bundle.model.chat(
        bundle.tokenizer,
        pixel_values,
        pixel_values_recons,
        question,
        diffusion_indices.to(device),
        generation_config=generation_config,
    )
