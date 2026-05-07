"""
upload_to_hf.py
────────────────
Uploads your collected dataset and trained LoRA adapter to Hugging Face Hub.
Required for project submission (links must appear in the report).

Setup:
  pip install huggingface_hub datasets
  huggingface-cli login   # paste your write token

Usage:
  python src/upload_to_hf.py --action dataset --metadata data/metadata.csv \
         --data_dir data --hf_username YOUR_USERNAME

  python src/upload_to_hf.py --action model --lora_path checkpoints/lora_final \
         --hf_username YOUR_USERNAME
"""

import os
import argparse
import pandas as pd
from datasets import Dataset, Image as HFImage
from huggingface_hub import HfApi


def upload_dataset(metadata_csv, data_dir, hf_username, dataset_name='penn-campus-time-of-day'):
    df = pd.read_csv(metadata_csv)
    df['image_path'] = df['filename'].apply(lambda f: os.path.join(data_dir, f))
    df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

    print(f'Uploading {len(df)} images to {hf_username}/{dataset_name}...')

    cols = {
        'image':       df['image_path'].tolist(),
        'location_id': df['location_id'].tolist(),
        'time':        df['time'].tolist(),
    }
    if 'lat' in df.columns:
        cols['lat'] = df['lat'].tolist()
        cols['lon'] = df['lon'].tolist()

    ds = Dataset.from_dict(cols).cast_column('image', HFImage())
    ds.push_to_hub(f'{hf_username}/{dataset_name}')

    print(f'Done: https://huggingface.co/datasets/{hf_username}/{dataset_name}')


def upload_model(lora_path, hf_username, model_name='penn-campus-pix2pix-lora'):
    api = HfApi()
    repo_id = f'{hf_username}/{model_name}'

    print(f'Creating repo {repo_id}...')
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type='model')

    # Add a model card before pushing
    card = f"""---
library_name: peft
base_model: timbrooks/instruct-pix2pix
tags:
- diffusers
- lora
- image-to-image
- style-transfer
---

# Penn Campus Time-of-Day Style Transfer

LoRA adapter for InstructPix2Pix, fine-tuned to perform time-of-day style
transfer on University of Pennsylvania campus scenes.

## Usage

```python
from diffusers import StableDiffusionInstructPix2PixPipeline
from peft import PeftModel
import torch

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to("cuda")
pipe.unet = PeftModel.from_pretrained(pipe.unet, "{repo_id}")
pipe.unet = pipe.unet.merge_and_unload()

result = pipe(
    prompt="turn this into a night scene with dark sky and warm window glow",
    image=source_image,
    num_inference_steps=30,
    image_guidance_scale=1.6,
    guidance_scale=7.5,
).images[0]
```

## Training Data

Trained on paired images from Penn campus locations under different times of day
(dawn, morning, noon, evening, night). See the companion dataset:
`{hf_username}/penn-campus-time-of-day`.

## Course Project

Built for CIS 4190/5190 Applied Machine Learning (Spring 2026) Final Project, Track C.
"""
    with open(os.path.join(lora_path, 'README.md'), 'w') as f:
        f.write(card)

    api.upload_folder(
        folder_path=lora_path,
        repo_id=repo_id,
        repo_type='model',
    )
    print(f'Done: https://huggingface.co/{repo_id}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--action',      required=True, choices=['dataset', 'model'])
    p.add_argument('--hf_username', required=True)
    p.add_argument('--metadata',    help='for dataset')
    p.add_argument('--data_dir',    help='for dataset')
    p.add_argument('--lora_path',   help='for model')
    args = p.parse_args()

    if args.action == 'dataset':
        upload_dataset(args.metadata, args.data_dir, args.hf_username)
    else:
        upload_model(args.lora_path, args.hf_username)
