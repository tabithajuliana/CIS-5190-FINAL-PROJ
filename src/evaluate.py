"""
evaluate.py
────────────
Reports the official + supplementary metrics required by the project spec:
  - LPIPS           (primary, lower is better)
  - Condition Acc   (primary, higher is better) -- via CLIP zero-shot classifier
  - SSIM            (supplementary)
  - PSNR            (supplementary)
  - CLIP score      (supplementary, image-condition alignment)

Runs over the released Validation_Pairs/ directory. Tests both directions of
each pair (e.g. morning -> night and night -> morning) and saves per-pair
results to CSV plus side-by-side qualitative figures.
"""

import os
import argparse
import torch
import lpips
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from transformers import CLIPProcessor, CLIPModel

from inference import StyleTransferModel, TIME_PROMPTS


# CLIP zero-shot text descriptions for each time-of-day class
CLIP_DESCRIPTIONS = {
    'dawn':    'a photograph at dawn with soft pink and orange light',
    'morning': 'a photograph in the morning with clear daylight',
    'noon':    'a photograph at midday with bright overhead sunlight',
    'evening': 'a photograph at golden hour with warm orange evening light',
    'night':   'a photograph at night with dark sky and artificial lighting',
}

TIME_LABELS = list(CLIP_DESCRIPTIONS.keys())


def parse_time_from_filename(fname):
    """Extract the time label from filenames like 'morning.jpg' or 'image5_dawn.jpg'."""
    name = os.path.splitext(fname)[0].lower()
    for t in TIME_LABELS:
        if t in name:
            return t
    return None


def load_pil_resized(path, size=512):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    left = (img.width - size) // 2
    top  = (img.height - size) // 2
    return img.crop((left, top, left+size, top+size))


class Evaluator:
    """Collects all metric models on a single device."""

    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)

        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        self.clip_proc  = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model.eval()

        self.lpips_t = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    @torch.no_grad()
    def lpips(self, generated, target):
        """Perceptual distance between two PIL images."""
        g = self.lpips_t(generated).unsqueeze(0).to(self.device)
        t = self.lpips_t(target.resize((512, 512))).unsqueeze(0).to(self.device)
        return self.lpips_fn(g, t).item()

    @torch.no_grad()
    def clip_classify(self, image):
        """Zero-shot classify the time-of-day of a generated image. Returns label string."""
        descriptions = list(CLIP_DESCRIPTIONS.values())
        inputs = self.clip_proc(
            text=descriptions, images=image,
            return_tensors='pt', padding=True
        ).to(self.device)
        logits = self.clip_model(**inputs).logits_per_image
        idx = int(logits.argmax(dim=1).item())
        return TIME_LABELS[idx]

    @torch.no_grad()
    def clip_score(self, image, target_time):
        """Cosine similarity between image and target description (higher = better aligned)."""
        target_text = CLIP_DESCRIPTIONS[target_time]
        inputs = self.clip_proc(
            text=[target_text], images=image,
            return_tensors='pt', padding=True
        ).to(self.device)
        outputs = self.clip_model(**inputs)
        # cosine similarity is logits_per_image divided by CLIP's logit_scale
        sim = outputs.logits_per_image.item() / self.clip_model.logit_scale.exp().item()
        return sim

    @staticmethod
    def ssim(generated, target):
        g = np.array(generated.resize((512, 512)))
        t = np.array(target.resize((512, 512)))
        return ssim_fn(g, t, channel_axis=2, data_range=255)

    @staticmethod
    def psnr(generated, target):
        g = np.array(generated.resize((512, 512)))
        t = np.array(target.resize((512, 512)))
        return psnr_fn(t, g, data_range=255)


def save_qualitative_figure(src_img, gen_img, target_img, target_time,
                             pair_id, src_time, save_dir):
    """Save a 3-panel comparison: source, generated, ground-truth target."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(src_img);    axes[0].set_title(f'Source ({src_time})',     fontsize=12)
    axes[1].imshow(gen_img);    axes[1].set_title(f'Generated ({target_time})', fontsize=12)
    axes[2].imshow(target_img); axes[2].set_title(f'Ground truth ({target_time})', fontsize=12)
    for a in axes: a.axis('off')
    plt.tight_layout()
    fname = f'pair{pair_id}_{src_time}_to_{target_time}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir',    required=True, help='Validation_Pairs directory')
    parser.add_argument('--lora_path',  default=None,  help='Path to LoRA adapter (or None for zero-shot)')
    parser.add_argument('--output',     required=True, help='Output CSV path')
    parser.add_argument('--save_qualitative', default=None,
                        help='If set, save side-by-side comparison figures here')
    parser.add_argument('--use_lora', type=str, default='true', choices=['true','false'])
    args = parser.parse_args()

    use_lora = args.use_lora.lower() == 'true' and args.lora_path is not None

    print(f'Mode: {"LoRA fine-tuned" if use_lora else "Zero-shot baseline"}')
    print(f'Loading model...')
    model = StyleTransferModel(lora_path=args.lora_path if use_lora else None)
    evaluator = Evaluator()

    pair_dirs = sorted(
        [d for d in os.listdir(args.val_dir)
         if os.path.isdir(os.path.join(args.val_dir, d)) and d.isdigit()],
        key=int
    )
    print(f'Found {len(pair_dirs)} validation pairs.\n')

    rows = []
    for pair_id in pair_dirs:
        pdir = os.path.join(args.val_dir, pair_id)
        files = [f for f in os.listdir(pdir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                 and not f.startswith('.')]
        if len(files) != 2:
            continue

        labeled = [(f, parse_time_from_filename(f)) for f in files]
        if any(t is None for _, t in labeled):
            print(f'Skipping pair {pair_id}: time labels not parseable')
            continue

        # Test both directions
        for src_file, src_time in labeled:
            tgt_file, tgt_time = next((f, t) for f, t in labeled if f != src_file)
            src_path = os.path.join(pdir, src_file)
            tgt_path = os.path.join(pdir, tgt_file)

            target_img = load_pil_resized(tgt_path)

            # Generate
            generated = model.transfer(src_path, tgt_time)

            # Compute metrics
            lp  = evaluator.lpips(generated, target_img)
            ss  = evaluator.ssim(generated, target_img)
            ps  = evaluator.psnr(generated, target_img)
            pred_time = evaluator.clip_classify(generated)
            cond_acc  = int(pred_time == tgt_time)
            cs        = evaluator.clip_score(generated, tgt_time)

            rows.append({
                'pair_id': pair_id,
                'direction': f'{src_time}_to_{tgt_time}',
                'lpips': round(lp, 4),
                'ssim':  round(ss, 4),
                'psnr':  round(ps, 2),
                'condition_correct': cond_acc,
                'predicted_time':    pred_time,
                'clip_score':        round(cs, 4),
            })
            print(f'  Pair {pair_id} {src_time}->{tgt_time}: '
                  f'LPIPS={lp:.4f} SSIM={ss:.4f} PSNR={ps:.2f} '
                  f'cond_correct={cond_acc} pred={pred_time}')

            # Save qualitative figure
            if args.save_qualitative:
                src_img = load_pil_resized(src_path)
                save_qualitative_figure(
                    src_img, generated, target_img,
                    target_time=tgt_time, pair_id=pair_id,
                    src_time=src_time, save_dir=args.save_qualitative
                )

    # Aggregate
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print('\n' + '='*60)
    print('FINAL METRICS')
    print('='*60)
    print(f'Mean LPIPS:        {df.lpips.mean():.4f}  (lower is better)')
    print(f'Condition Acc:     {df.condition_correct.mean():.4f}  (higher is better)')
    print(f'Mean SSIM:         {df.ssim.mean():.4f}')
    print(f'Mean PSNR:         {df.psnr.mean():.2f}')
    print(f'Mean CLIP score:   {df.clip_score.mean():.4f}')
    print(f'\nResults saved to {args.output}')

    # Per-direction breakdown
    print('\nBy direction:')
    print(df.groupby('direction')[
        ['lpips','ssim','psnr','condition_correct']
    ].mean().round(4))


if __name__ == '__main__':
    main()
