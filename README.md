# eXpress CVPR InternVL

This repository is the cleaned implementation for the CVPR 2026 project.

It keeps:

- the local InternVL chat model implementation with reconstruction-guided generation;
- Stable Diffusion based image reconstruction utilities;
- one runnable sample for "reconstruct image + answer question".

It does not keep experiment outputs, images, spreadsheets, notebooks, checkpoints, PDFs, or LLaVA/Qwen comparison code.

## Environment

Create the requested conda environment:

```bash
conda create -n cvpr2026 python=3.10
conda activate cvpr2026
pip install -r requirements.txt
pip install -e .
```

If you prefer an environment file:

```bash
conda env create -f environment.yml
conda activate cvpr2026
pip install -e .
```

## Model Weights

Model weights are not included. Use local paths or Hugging Face model ids.

For local weights:

```bash
export INTERNVL_MODEL=/data/model_weights/InternVL3_5-8B
export SD_MODEL=/data/model_weights/stable-diffusion-v1-5
```

## Run The Sample

```bash
python examples/reconstruct_and_answer.py \
  --image /path/to/image.jpg \
  --question "Count the legs of this animal. Please answer with a single number." \
  --internvl-model "$INTERNVL_MODEL" \
  --sd-model "$SD_MODEL" \
  --local-files-only
```

Optional outputs:

```bash
python examples/reconstruct_and_answer.py \
  --image /path/to/image.jpg \
  --question "What is unusual in this image?" \
  --save-reconstruction outputs/reconstructed.png \
  --save-uncertainty outputs/uncertainty.npy
```

`outputs/` is ignored by git.

## Layout

```text
internvl/                 Local InternVL model code
src/express_cvpr/         Reconstruction and InternVL runner utilities
examples/                 Minimal runnable test sample
requirements.txt          Python dependencies
environment.yml           Conda environment name: cvpr2026
```

