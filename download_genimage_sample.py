"""Download and populate a mini sample dataset in GenImage format for immediate testing."""

import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import requests

def prepare_sample_dataset():
    sample_dir = "data/genimage_sample"
    categories = ["Real_ImageNet", "Midjourney_v5", "StableDiffusion_v1_5", "BigGAN", "Wukong"]

    print("=" * 70)
    print("DOWNLOADING & POPULATING SAMPLE GENIMAGE DATASET")
    print("=" * 70)

    for cat in categories:
        os.makedirs(os.path.join(sample_dir, cat), exist_ok=True)

    # Base test image URLs
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/encode_jpeg/grace_hopper.jpg",
        "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/exporter/astronaut.jpg",
    ]

    base_images = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for idx, url in enumerate(urls):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                temp_p = f"data/temp_{idx}.jpg"
                with open(temp_p, "wb") as f:
                    f.write(resp.content)
                img = Image.open(temp_p).convert("RGB")
                base_images.append(img)
                if os.path.exists(temp_p):
                    os.remove(temp_p)
        except Exception as e:
            print(f"Warning: Could not fetch {url}: {e}")

    if not base_images:
        print("Creating synthetic base fallback images...")
        for _ in range(3):
            arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
            base_images.append(Image.fromarray(arr))

    print(f"Loaded {len(base_images)} base source images. Generating generator-style variations...")

    num_samples_per_cat = 25

    for cat in categories:
        out_dir = os.path.join(sample_dir, cat)
        print(f"  • Populating '{cat}' ({num_samples_per_cat} images)...")

        for i in range(num_samples_per_cat):
            src_img = base_images[i % len(base_images)].copy()
            # Crop/Resize variation
            w, h = src_img.size
            crop_box = (i % 20, i % 20, w - (i % 20), h - (i % 20))
            img_variant = src_img.crop(crop_box).resize((224, 224), Image.Resampling.BICUBIC)

            # Apply generator-characteristic style modifications
            if cat == "Midjourney_v5":
                # High contrast, cinematic saturation
                img_variant = ImageEnhance.Contrast(img_variant).enhance(1.3 + (i % 5) * 0.05)
                img_variant = ImageEnhance.Color(img_variant).enhance(1.2)
            elif cat == "StableDiffusion_v1_5":
                # Subtle Gaussian blur and soft tone
                img_variant = img_variant.filter(ImageFilter.GaussianBlur(radius=0.5 + (i % 3) * 0.2))
            elif cat == "BigGAN":
                # Sharpness and slight color shift
                img_variant = ImageEnhance.Sharpness(img_variant).enhance(1.5)
            elif cat == "Wukong":
                # Warm color balance
                arr = np.array(img_variant, dtype=np.float32)
                arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.08, 0, 255)
                img_variant = Image.fromarray(arr.astype(np.uint8))

            img_variant.save(os.path.join(out_dir, f"{cat.lower()}_{i+1:03d}.jpg"))

    print("\n" + "=" * 70)
    print(f"SUCCESS: Sample GenImage dataset populated at '{sample_dir}'")
    print("=" * 70)

if __name__ == "__main__":
    prepare_sample_dataset()
