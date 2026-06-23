from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from scipy.ndimage import zoom
from torchvision import transforms


@dataclass
class ReconstructionResult:
    image: Image.Image
    uncertainty_map: np.ndarray
    diffusion_indices: torch.LongTensor


def pipeline_device(pipeline: StableDiffusionPipeline) -> torch.device:
    return getattr(pipeline, "_execution_device", getattr(pipeline, "device", torch.device("cpu")))


def load_diffusion_pipeline(
    model_id: str,
    device: torch.device | str,
    torch_dtype: torch.dtype | None = None,
    local_files_only: bool = False,
) -> StableDiffusionPipeline:
    if torch_dtype is None:
        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def preprocess_pillow_image(
    image: Image.Image,
    image_size: tuple[int, int] = (512, 512),
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    preprocess = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    return preprocess(image.convert("RGB")).unsqueeze(0).to(device)


@torch.no_grad()
def get_conditional_embeddings(
    pipeline: StableDiffusionPipeline,
    prompt: str = "",
) -> torch.Tensor:
    pred = pipeline.encode_prompt(
        prompt,
        device=pipeline_device(pipeline),
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return pred[0].to(pipeline.unet.dtype)


@torch.no_grad()
def encode(pipeline: StableDiffusionPipeline, image: torch.Tensor) -> torch.Tensor:
    image = image.to(pipeline.vae.dtype) * 2 - 1
    latent = pipeline.vae.encode(image).latent_dist.mode() * pipeline.vae.config.scaling_factor
    return latent.to(pipeline.unet.dtype)


@torch.no_grad()
def decode(pipeline: StableDiffusionPipeline, latent: torch.Tensor) -> torch.Tensor:
    latent = latent.to(pipeline.vae.dtype)
    return pipeline.vae.decode(latent / pipeline.vae.config.scaling_factor).sample


@torch.no_grad()
def normalize_tensor(tensor: torch.Tensor, dim: Sequence[int] | None = None) -> torch.Tensor:
    if dim is None:
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-20)

    tensor_min = tensor
    tensor_max = tensor
    for d in reversed(dim):
        tensor_min = torch.min(tensor_min, d, keepdim=True)[0]
        tensor_max = torch.max(tensor_max, d, keepdim=True)[0]
    return (tensor - tensor_min) / (tensor_max - tensor_min + 1e-20)


def to01_img(tensor_bchw: torch.Tensor) -> np.ndarray:
    arr = tensor_bchw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


@torch.no_grad()
def latent_refine_batch(
    pipeline: StableDiffusionPipeline,
    z: torch.Tensor,
    t: torch.Tensor,
    text_embeds: torch.Tensor,
    lr_range: tuple[float, float] = (0.1, 0.0),
    num_anneal: int = 10,
    drift_lambda: float = 0.0,
) -> torch.Tensor:
    x = z.clone()
    alphas = pipeline.scheduler.alphas_cumprod.to(device=z.device, dtype=z.dtype)
    sigma_t = (1 - alphas[t]) ** 0.5
    for i in range(num_anneal):
        lr = min(i, num_anneal - 1) / max(num_anneal - 1, 1) * (lr_range[1] - lr_range[0]) + lr_range[0]
        eta = lr**2
        eps_pred = pipeline.unet(x, t, encoder_hidden_states=text_embeds).sample
        score = -eps_pred / sigma_t[:, None, None, None]
        x = x + eta * score + (2 * eta) ** 0.5 * torch.randn_like(x)
        x = (1 - drift_lambda) * x + drift_lambda * z
    return x


@torch.no_grad()
def ddim_partial_inversion_batch(
    pipeline: StableDiffusionPipeline,
    z_start: torch.Tensor,
    inv_depths: Sequence[int],
    text_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    timesteps = pipeline.scheduler.timesteps
    alphas = pipeline.scheduler.alphas_cumprod.to(device=z_start.device, dtype=z_start.dtype)
    inv_depths = sorted(int(depth) for depth in inv_depths)
    max_depth = max(inv_depths)
    reverse_timesteps = torch.flip(timesteps, dims=[0])
    candidate_timesteps = reverse_timesteps[torch.tensor(inv_depths, device=timesteps.device)]
    sub_timesteps = timesteps[-(max_depth + 1) :]

    z_t = z_start.clone()
    z_list = []
    candidate_set = {int(t.item()) for t in candidate_timesteps}
    for i in range(len(sub_timesteps) - 1, 0, -1):
        t = int(sub_timesteps[i].item())
        t_prev = int(sub_timesteps[i - 1].item())
        eps = pipeline.unet(z_t, t, encoder_hidden_states=text_embeds).sample
        a_t = alphas[t]
        a_prev = alphas[t_prev]
        x0_hat = (z_t - (1 - a_t).sqrt() * eps) / (a_t.sqrt() + 1e-8)
        z_t = a_prev.sqrt() * x0_hat + (1 - a_prev).sqrt() * eps
        if t_prev in candidate_set:
            z_list.append(z_t.clone())

    if len(z_list) != len(inv_depths):
        raise RuntimeError("Failed to collect all requested inversion depths.")
    return sub_timesteps, torch.stack(z_list, 0), candidate_timesteps


@torch.no_grad()
def ddim_partial_reconstruct_batch(
    pipeline: StableDiffusionPipeline,
    z_tstar: torch.Tensor,
    candidate_timesteps: torch.Tensor,
    sub_timesteps: torch.Tensor,
    text_embeds: torch.Tensor,
) -> torch.Tensor:
    batch_size, samples, channels, height, width = z_tstar.shape
    z_t = z_tstar.new_zeros(z_tstar.size())
    candidate_lookup = {int(t.item()): idx for idx, t in enumerate(candidate_timesteps)}

    for t in sub_timesteps:
        t_int = int(t.item())
        if t_int in candidate_lookup:
            idx = candidate_lookup[t_int]
            z_t[idx] = z_tstar[idx].clone()

        z_t_flat = z_t.flatten(0, 1)
        eps = pipeline.unet(z_t_flat, t_int, encoder_hidden_states=text_embeds).sample
        step_out = pipeline.scheduler.step(eps, t_int, z_t_flat)
        z_t_flat = step_out["prev_sample"] if isinstance(step_out, dict) else step_out.prev_sample
        z_t = z_t_flat.reshape(batch_size, samples, channels, height, width)

    return z_t


@torch.no_grad()
def reconstruct_candidates(
    pipeline: StableDiffusionPipeline,
    image: torch.Tensor,
    num_steps: int,
    inv_depths: Sequence[int],
    samples_per_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uncond = get_conditional_embeddings(pipeline, "")
    z0 = encode(pipeline, image)
    _, channels, height, width = z0.shape
    num_depths = len(inv_depths)

    pipeline.scheduler.set_timesteps(num_steps, device=z0.device)
    sub_timesteps, z_t, candidate_timesteps = ddim_partial_inversion_batch(pipeline, z0, inv_depths, uncond)

    z_t = z_t.expand(-1, samples_per_depth, -1, -1, -1).flatten(0, 1)
    z_times = candidate_timesteps[:, None].expand(-1, samples_per_depth).flatten(0, 1).to(z_t.device)
    uncond = uncond[None].expand(num_depths, samples_per_depth, -1, -1).flatten(0, 1)

    z_t = latent_refine_batch(pipeline, z_t, z_times, text_embeds=uncond)
    z_t = z_t.reshape(num_depths, samples_per_depth, channels, height, width)
    z0_rec = ddim_partial_reconstruct_batch(pipeline, z_t, candidate_timesteps, sub_timesteps, uncond)

    sort_idx = (z0_rec - z0[None]).pow(2).reshape(num_depths, samples_per_depth, -1).sum(-1).argsort(dim=1)
    median_z0_rec = torch.gather(
        z0_rec,
        1,
        sort_idx[..., None, None, None].expand(-1, -1, channels, height, width),
    )[:, samples_per_depth // 2]

    z0_rec_norm = normalize_tensor(z0_rec, dim=[2, 3, 4])
    z0_norm = normalize_tensor(z0, dim=[1, 2, 3])
    uncertainty_map = (z0_norm[None] - z0_rec_norm).pow(2).mean([1, 2])
    uncertainty_map = normalize_tensor(uncertainty_map, dim=[1, 2]).detach().cpu().numpy().astype(np.float32)
    uncertainty_map = np.repeat(uncertainty_map, 8, axis=1)
    uncertainty_map = np.repeat(uncertainty_map, 8, axis=2)

    input_np = to01_img(image)
    image_rec = decode(pipeline, median_z0_rec)
    image_rec = (image_rec / 2 + 0.5).clamp(0, 1)
    rec_np = np.stack([to01_img(img[None]) for img in image_rec], axis=0)

    pixel_diff = ((input_np - rec_np) ** 2).reshape(num_depths, -1).mean(1)
    pixel_diff = (pixel_diff - pixel_diff.min()) / (pixel_diff.max() - pixel_diff.min() + 1e-20)
    latent_var = uncertainty_map.mean((1, 2))
    latent_var = (latent_var - latent_var.min()) / (latent_var.max() - latent_var.min() + 1e-20)

    return input_np, rec_np, uncertainty_map, pixel_diff, latent_var


def select_diffusion_indices(
    uncertainty_map: np.ndarray,
    target_aspect_ratio: tuple[int, int],
    tile_topk: int = 64,
    thumbnail_topk: int = 32,
) -> torch.LongTensor:
    cols, rows = target_aspect_ratio
    global_map = zoom(uncertainty_map, (16 / uncertainty_map.shape[0], 16 / uncertainty_map.shape[1]), order=1)
    tile_map = zoom(
        uncertainty_map,
        (16 * rows / uncertainty_map.shape[0], 16 * cols / uncertainty_map.shape[1]),
        order=1,
    )

    patch_h = tile_map.shape[0] // rows
    patch_w = tile_map.shape[1] // cols
    patches = tile_map.reshape(rows, patch_h, cols, patch_w).transpose(0, 2, 1, 3)
    patches = patches.reshape(rows * cols, patch_h * patch_w)

    tile_scores = torch.from_numpy(patches).flatten()
    global_scores = torch.from_numpy(global_map).flatten()
    tile_topk = min(tile_topk, tile_scores.numel())
    thumbnail_topk = min(thumbnail_topk, global_scores.numel())

    tile_indices = torch.topk(tile_scores, tile_topk, largest=True).indices
    global_indices = torch.topk(global_scores, thumbnail_topk, largest=True).indices
    global_indices = global_indices + 256 * cols * rows
    return torch.cat([tile_indices, global_indices], dim=0).long().sort().values


@torch.no_grad()
def reconstruct_for_internvl(
    pipeline: StableDiffusionPipeline,
    image: Image.Image,
    target_aspect_ratio: tuple[int, int],
    num_steps: int = 50,
    inv_depths: Sequence[int] = (15,),
    samples_per_depth: int = 8,
    diffusion_image_size: tuple[int, int] = (512, 512),
    tile_topk: int = 64,
    thumbnail_topk: int = 32,
) -> ReconstructionResult:
    tensor_image = preprocess_pillow_image(image, image_size=diffusion_image_size, device=pipeline_device(pipeline))
    _, rec_np, uncertainty_maps, pixel_diff, latent_var = reconstruct_candidates(
        pipeline,
        tensor_image,
        num_steps=num_steps,
        inv_depths=inv_depths,
        samples_per_depth=samples_per_depth,
    )
    target_idx = int((0.5 * pixel_diff + latent_var).argmin())
    reconstructed = transforms.ToPILImage()(
        torch.from_numpy(rec_np[target_idx]).permute(2, 0, 1).to(torch.float32)
    )
    reconstructed = reconstructed.resize(image.size, Image.LANCZOS)
    uncertainty_map = uncertainty_maps[target_idx]
    diffusion_indices = select_diffusion_indices(
        uncertainty_map,
        target_aspect_ratio=target_aspect_ratio,
        tile_topk=tile_topk,
        thumbnail_topk=thumbnail_topk,
    )
    return ReconstructionResult(reconstructed, uncertainty_map, diffusion_indices)
