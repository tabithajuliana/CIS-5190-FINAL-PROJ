"""
train_lora.py
──────────────
LoRA fine-tuning of InstructPix2Pix on Penn campus paired images.

Usage:
    python src/train_lora.py \
        --data_dir data \
        --metadata data/metadata.csv \
        --epochs 5 \
        --output checkpoints/lora_final
"""

import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionInstructPix2PixPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

from inference import TIME_PROMPTS


class CampusPairedDataset(Dataset):
    """Builds (source, target, prompt) triplets from paired campus images."""

    def __init__(self, csv_path, img_dir, size=512):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['filename', 'location_id', 'time'])
        df['time'] = df['time'].astype(str).str.strip().str.lower()
        df = df[~df['time'].isin(['', 'nan'])].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.pairs   = []
        self.prompts = []

        for loc_id, group in df.groupby('location_id'):
            rows = group.to_dict('records')
            for src in rows:
                for tgt in rows:
                    if src['filename'] == tgt['filename']:
                        continue
                    if src['time'] == tgt['time']:
                        continue
                    if tgt['time'] not in TIME_PROMPTS:
                        continue
                    src_path = os.path.join(img_dir, src['filename'])
                    tgt_path = os.path.join(img_dir, tgt['filename'])
                    if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
                        continue
                    self.pairs.append((src_path, tgt_path))
                    self.prompts.append(TIME_PROMPTS[tgt['time']])

        if not self.pairs:
            raise ValueError('No training pairs found. Check metadata.csv.')

        print(f'Built dataset: {len(self.pairs)} pairs across {df.location_id.nunique()} locations')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src_path, tgt_path = self.pairs[i]
        return {
            'source': self.transform(Image.open(src_path).convert('RGB')),
            'target': self.transform(Image.open(tgt_path).convert('RGB')),
            'prompt': self.prompts[i],
        }


def train(args):
    device = 'cuda'

    # 1. Load base model
    print('Loading base InstructPix2Pix model...')
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix',
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()

    # 2. Attach LoRA adapter
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
        lora_dropout=0.05,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    # Cast LoRA params to fp32 to prevent fp16 overflow during training
    for name, p in pipe.unet.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # 3. Setup training
    noise_scheduler = DDPMScheduler.from_pretrained(
        'timbrooks/instruct-pix2pix', subfolder='scheduler'
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()),
        lr=args.lr
    )

    dataset = CampusPairedDataset(args.metadata, args.data_dir)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    pipe.unet.train()
    loss_history = []
    scaler = torch.cuda.amp.GradScaler()

    print(f'\nStarting training: {args.epochs} epochs x {len(loader)} steps')

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            tgt    = batch['target'].to(device, dtype=torch.float16)
            src    = batch['source'].to(device, dtype=torch.float16)
            prompt = batch['prompt']

            with torch.no_grad():
                tgt_lat = pipe.vae.encode(tgt).latent_dist.sample() * 0.18215
                src_lat = pipe.vae.encode(src).latent_dist.sample() * 0.18215
                text_ids = pipe.tokenizer(
                    prompt, return_tensors='pt',
                    padding='max_length', truncation=True, max_length=77
                ).input_ids.to(device)
                text_emb = pipe.text_encoder(text_ids)[0]

            noise = torch.randn_like(tgt_lat)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (tgt_lat.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(tgt_lat, noise, t)

            model_input = torch.cat([noisy, src_lat], dim=1)
            with torch.autocast('cuda', dtype=torch.float16):
                pred = pipe.unet(model_input, t, encoder_hidden_states=text_emb).sample
            loss = F.mse_loss(pred.float(), noise.float())

            if torch.isnan(loss) or torch.isinf(loss):
                print(f'    Skipping step {step}: loss={loss.item()}')
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipe.unet.parameters() if p.requires_grad],
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f'  Epoch {epoch+1}/{args.epochs} step {step}/{len(loader)} '
                      f'loss={loss.item():.4f}')

        avg = epoch_loss / len(loader)
        loss_history.append(avg)
        ckpt = f'{args.output}_epoch{epoch+1}'
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        pipe.unet.save_pretrained(ckpt)
        print(f'  Epoch {epoch+1} avg loss={avg:.4f}, saved {ckpt}')

    # Final
    pipe.unet.save_pretrained(args.output)
    print(f'\nFinal LoRA saved to {args.output}')

    # Save loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('LoRA Fine-tuning Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.output}_loss.png', dpi=150)
    print(f'Loss curve saved to {args.output}_loss.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--epochs',   type=int, default=5)
    p.add_argument('--lr',       type=float, default=1e-4)
    p.add_argument('--rank',     type=int, default=4)
    p.add_argument('--output',   default='checkpoints/lora_final')
    args = p.parse_args()
    train(args)
