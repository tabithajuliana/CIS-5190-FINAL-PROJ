"""
classical_baseline.py
──────────────────────
Naive pixel-based time-of-day transfer using HSV color manipulation.
Used in the report as a baseline to demonstrate why a generative model is needed.

This baseline applies fixed color/lightness curves per target time-of-day. It
serves as the "no-ML" lower bound for the comparison exploratory component.

Usage:
    python src/classical_baseline.py \
        --val_dir Validation_Pairs \
        --output outputs/classical_metrics.csv \
        --save_qualitative outputs/qualitative_classical
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


# Per-time color/lightness adjustments derived from inspecting validation pairs
# Format: (R_mult, G_mult, B_mult, brightness, saturation, blue_sky_blend)
TIME_FILTERS = {
    'dawn':    (1.10, 0.92, 0.82, 0.90, 1.05, 0.10),  # warm pink-orange, slightly dim
    'morning': (1.00, 1.00, 1.00, 1.00, 1.00, 0.00),  # neutral reference
    'noon':    (1.05, 1.02, 0.95, 1.10, 1.10, 0.05),  # bright, slightly warm
    'evening': (1.18, 0.95, 0.75, 0.85, 1.20, 0.00),  # strong warm/golden cast
    'night':   (0.55, 0.55, 0.75, 0.45, 0.70, -0.30), # dark, slightly blue
}


def classical_transfer(image, target_time):
    """Apply fixed pixel-level filter for the target time-of-day."""
    if target_time not in TIME_FILTERS:
        return image

    rm, gm, bm, brightness, sat, sky = TIME_FILTERS[target_time]

    arr = np.array(image, dtype=np.float32)
    arr[..., 0] *= rm
    arr[..., 1] *= gm
    arr[..., 2] *= bm

    # Blue sky region adjustment (top 30% of image)
    if abs(sky) > 0.01:
        h = arr.shape[0]
        sky_zone = arr[:int(h*0.3), :, :]
        if sky > 0:
            sky_zone[..., 2] = np.clip(sky_zone[..., 2] * (1 + sky) + 30 * sky, 0, 255)
        else:
            sky_zone[...] *= (1 + sky)
        arr[:int(h*0.3), :, :] = sky_zone

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Color(out).enhance(sat)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir',  required=True)
    parser.add_argument('--output',   required=True)
    parser.add_argument('--save_qualitative', default=None)
    args = parser.parse_args()

    # Lazy imports to avoid loading heavy ML libs unnecessarily
    import torch
    import lpips
    from torchvision import transforms
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn

    from evaluate import parse_time_from_filename, load_pil_resized

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    to_t = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    if args.save_qualitative:
        os.makedirs(args.save_qualitative, exist_ok=True)

    pair_dirs = sorted(
        [d for d in os.listdir(args.val_dir)
         if os.path.isdir(os.path.join(args.val_dir, d)) and d.isdigit()],
        key=int
    )

    rows = []
    for pid in pair_dirs:
        pdir = os.path.join(args.val_dir, pid)
        files = [f for f in os.listdir(pdir)
                 if f.lower().endswith(('.jpg','.jpeg','.png')) and not f.startswith('.')]
        if len(files) != 2:
            continue
        labeled = [(f, parse_time_from_filename(f)) for f in files]
        if any(t is None for _, t in labeled):
            continue

        for src_file, src_time in labeled:
            tgt_file, tgt_time = next((f, t) for f, t in labeled if f != src_file)
            src_img = load_pil_resized(os.path.join(pdir, src_file))
            tgt_img = load_pil_resized(os.path.join(pdir, tgt_file))

            generated = classical_transfer(src_img, tgt_time)

            with torch.no_grad():
                g = to_t(generated).unsqueeze(0).to(device)
                t = to_t(tgt_img).unsqueeze(0).to(device)
                lp = lpips_fn(g, t).item()
            ss = ssim_fn(np.array(generated), np.array(tgt_img),
                         channel_axis=2, data_range=255)
            ps = psnr_fn(np.array(tgt_img), np.array(generated), data_range=255)

            rows.append({
                'pair_id': pid,
                'direction': f'{src_time}_to_{tgt_time}',
                'lpips': round(lp, 4),
                'ssim':  round(ss, 4),
                'psnr':  round(ps, 2),
            })
            print(f'  Pair {pid} {src_time}->{tgt_time}: '
                  f'LPIPS={lp:.4f} SSIM={ss:.4f}')

            if args.save_qualitative:
                generated.save(
                    os.path.join(args.save_qualitative,
                                 f'pair{pid}_{src_time}_to_{tgt_time}_classical.jpg'),
                    quality=92
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print('\nClassical baseline aggregate:')
    print(f'  Mean LPIPS: {df.lpips.mean():.4f}  (lower is better)')
    print(f'  Mean SSIM:  {df.ssim.mean():.4f}')
    print(f'  Mean PSNR:  {df.psnr.mean():.2f}')
    print(f'\nResults: {args.output}')


if __name__ == '__main__':
    main()
