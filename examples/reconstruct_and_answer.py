from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from express_cvpr.internvl_runner import answer_with_reconstruction, load_internvl
from express_cvpr.preprocess import dynamic_preprocess
from express_cvpr.reconstruction import load_diffusion_pipeline, reconstruct_for_internvl


def parse_inv_depths(value: str) -> list[int]:
    depths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not depths:
        raise argparse.ArgumentTypeError("At least one inversion depth is required.")
    return depths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct an image and answer with InternVL.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--question", required=True, help="Question to ask InternVL.")
    parser.add_argument(
        "--internvl-model",
        default=os.environ.get("INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B"),
        help="InternVL model path or Hugging Face id.",
    )
    parser.add_argument(
        "--sd-model",
        default=os.environ.get("SD_MODEL", "runwayml/stable-diffusion-v1-5"),
        help="Stable Diffusion v1.5 model path or Hugging Face id.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--inv-depths", type=parse_inv_depths, default=parse_inv_depths("15"))
    parser.add_argument("--samples-per-depth", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--tile-topk", type=int, default=64)
    parser.add_argument("--thumbnail-topk", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--save-reconstruction", default=None, help="Optional output path for the reconstructed image.")
    parser.add_argument("--save-uncertainty", default=None, help="Optional output path for the uncertainty .npy file.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image = Image.open(args.image).convert("RGB")
    _, target_aspect_ratio = dynamic_preprocess(
        image,
        image_size=args.image_size,
        use_thumbnail=True,
        max_num=args.max_tiles,
    )

    sd_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = load_diffusion_pipeline(
        args.sd_model,
        device=device,
        torch_dtype=sd_dtype,
        local_files_only=args.local_files_only,
    )
    reconstruction = reconstruct_for_internvl(
        pipe,
        image,
        target_aspect_ratio=target_aspect_ratio,
        num_steps=args.num_steps,
        inv_depths=args.inv_depths,
        samples_per_depth=args.samples_per_depth,
        tile_topk=args.tile_topk,
        thumbnail_topk=args.thumbnail_topk,
    )

    if args.save_reconstruction:
        output_path = Path(args.save_reconstruction)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction.image.save(output_path)
    if args.save_uncertainty:
        output_path = Path(args.save_uncertainty)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, reconstruction.uncertainty_map)

    internvl_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    bundle = load_internvl(args.internvl_model, device=device, dtype=internvl_dtype, use_flash_attn=False)
    answer = answer_with_reconstruction(
        bundle,
        image,
        reconstruction.image,
        reconstruction.diffusion_indices,
        args.question,
        image_size=args.image_size,
        max_tiles=args.max_tiles,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
