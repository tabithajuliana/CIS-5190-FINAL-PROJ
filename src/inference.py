"""
inference.py
─────────────
Single-image style transfer using fine-tuned InstructPix2Pix + LoRA.

Usage:
    from src.inference import StyleTransferModel
    model = StyleTransferModel(lora_path="checkpoints/lora_final.pt")
    out = model.transfer("photo.jpg", target_time="night")
    out.save("photo_night.jpg")
"""

import torch
from PIL import Image
from pillow_heif import register_heif_opener
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from peft import PeftModel

register_heif_opener()  # enable iPhone .HEIC files

# Time-of-day prompt templates aligned with the validation set vocabulary
TIME_PROMPTS = {
    'dawn':    'turn this into a dawn scene with soft pink-orange light and warm glowing lamps',
    'morning': 'turn this into a bright morning scene with clear daylight and soft shadows',
    'noon':    'turn this into a midday scene with bright overhead sun and strong shadows',
    'evening': 'turn this into a golden hour evening scene with warm orange light',
    'night':   'turn this into a night scene with dark sky and warm window glow',
}


class StyleTransferModel:
    """Wraps the fine-tuned InstructPix2Pix pipeline for inference."""

    BASE_MODEL = 'timbrooks/instruct-pix2pix'

    def __init__(self, lora_path=None, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype  = dtype

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.BASE_MODEL,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        # Faster, better scheduler per the InstructPix2Pix paper
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_attention_slicing()

        # Load LoRA adapter if provided
        if lora_path:
            self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_path)
            self.pipe.unet = self.pipe.unet.merge_and_unload()  # bake LoRA into base
            print(f'Loaded LoRA adapter from {lora_path}')

    @staticmethod
    def _load_image(path, size=512):
        """Load any image format and resize to a square center crop."""
        img = Image.open(path).convert('RGB')
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        left = (img.width  - size) // 2
        top  = (img.height - size) // 2
        return img.crop((left, top, left+size, top+size))

    def transfer(self, image_path, target_time,
                 steps=30, image_guidance=2.5, text_guidance=5.0, seed=42):
        """
        Generate a time-shifted version of the input image.

        Args:
            image_path:      path to source .jpg / .png / .HEIC
            target_time:     one of 'dawn', 'morning', 'noon', 'evening', 'night'
            steps:           inference steps (20-50)
            image_guidance:  scene preservation (higher = closer to source)
            text_guidance:   prompt strength (higher = stronger time effect)
            seed:            for reproducibility

        Returns:
            PIL.Image (512x512)
        """
        if target_time not in TIME_PROMPTS:
            raise ValueError(
                f'Unknown target_time {target_time!r}. '
                f'Must be one of: {list(TIME_PROMPTS.keys())}'
            )

        prompt = TIME_PROMPTS[target_time]
        src_img = self._load_image(image_path)
        gen = torch.Generator(self.device).manual_seed(seed)

        with torch.autocast(self.device):
            out = self.pipe(
                prompt=prompt,
                image=src_img,
                num_inference_steps=steps,
                image_guidance_scale=image_guidance,
                guidance_scale=text_guidance,
                generator=gen,
            ).images[0]
        return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='path to input image')
    p.add_argument('--target_time', required=True, choices=list(TIME_PROMPTS.keys()))
    p.add_argument('--lora_path', default=None, help='path to LoRA adapter')
    p.add_argument('--output', default='output.jpg')
    args = p.parse_args()

    model = StyleTransferModel(lora_path=args.lora_path)
    result = model.transfer(args.image, args.target_time)
    result.save(args.output, quality=95)
    print(f'Saved -> {args.output}')
