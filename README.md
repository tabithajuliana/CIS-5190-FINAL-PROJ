# Penn Style Transfer - Method A (Paired LoRA)

Conditional time-of-day style transfer for University of Pennsylvania campus scenes,
using LoRA fine-tuning of Stable Diffusion InstructPix2Pix on paired training data.

CIS 4190/5190 final project, Track C. Spring 2026.

## Authors

Tabitha Appala, Pranay [LASTNAME], Sumanth [LASTNAME]

## Links

- Dataset (Hugging Face): [paste HF dataset URL after upload]
- Trained adapter (Hugging Face): [paste HF model URL after upload]
- Method B repository: [link to companion repo]

## Repository structure

```
.
├── src/
│   ├── train_lora.py          # LoRA fine-tuning entry point
│   ├── inference.py           # single-image style transfer
│   ├── evaluate.py            # LPIPS, CLIP CondAcc, SSIM, PSNR
│   ├── classical_baseline.py  # pixel-based comparison baseline
│   ├── prepare_data.py        # builds metadata.csv from photos
│   └── upload_to_hf.py        # publishes to Hugging Face
├── run_pipeline.ipynb         # end-to-end Colab pipeline
├── figures/                   # qualitative comparison figures
├── metrics/                   # per-pair and aggregate metric CSVs
├── requirements.txt
└── README.md
```

The trained adapter checkpoint and the photo dataset are too large for git;
both are released on Hugging Face Hub.

## Reproducing the results

Tested on Google Colab with a T4 GPU and Python 3.12.

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Build dataset metadata

```bash
python src/prepare_data.py --data_dir data --output data/metadata.csv
```

### 3. Zero-shot baseline

```bash
python src/evaluate.py \
    --val_dir Validation_Pairs --use_lora false \
    --output outputs/zeroshot_metrics.csv \
    --save_qualitative outputs/qualitative_zeroshot
```

Expected runtime: ~12 min on T4. Expected mean LPIPS: ~0.64.

### 4. LoRA fine-tuning

```bash
python src/train_lora.py \
    --data_dir data --metadata data/metadata.csv \
    --epochs 5 --lr 1e-4 --rank 4 \
    --output checkpoints/lora_final
```

### 5. Final evaluation

```bash
python src/evaluate.py \
    --val_dir Validation_Pairs --use_lora true \
    --lora_path checkpoints/lora_final \
    --output outputs/finetuned_metrics.csv \
    --save_qualitative outputs/qualitative_finetuned
```

### 6. Single-image inference

```python
from src.inference import StyleTransferModel
model = StyleTransferModel(lora_path="checkpoints/lora_final")
out = model.transfer("photo.jpg", target_time="night")
out.save("photo_night.jpg")
```

## Method summary

We fine-tune the pretrained timbrooks/instruct-pix2pix model with LoRA adapters
attached to the U-Net attention projections (rank 4, alpha 4). Training uses paired
(source, target) images from the same campus location across different times of day,
with MSE loss on predicted noise. The VAE and text encoder are frozen. LoRA
parameters are stored in fp32 for numerical stability, and we use mixed-precision
training with GradScaler and gradient clipping. See the project report for full
methodology and the comparison against Method B.

## Headline results (validation set, 42 samples)

| Method | LPIPS (lower) | CondAcc (higher) | SSIM | PSNR |
|---|---|---|---|---|
| Classical (HSV manipulation) | 0.624 | -- | 0.138 | 11.44 |
| Zero-shot InstructPix2Pix | 0.639 | 0.476 | 0.129 | 11.17 |
| Method A: paired LoRA (ours) | 0.772 | **0.667** | 0.132 | 10.69 |

See `metrics/method_comparison.csv` for full numbers.

## License

Code: MIT. Dataset: CC-BY-NC 4.0 (educational use only).
