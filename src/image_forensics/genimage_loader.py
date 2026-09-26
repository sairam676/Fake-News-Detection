"""GenImage Dataset Loader and Multi-Model Feature Caching Utility.

Supports standard GenImage subfolder formats:
  - imagenet_ai_0508_midjourney
  - imagenet_ai_0508_sdv1_4
  - imagenet_ai_0508_sdv1_5
  - imagenet_ai_0508_biggan
  - imagenet_ai_0508_wukong
  - imagenet_ai_0508_glide
  - imagenet_ai_0508_vqdm
  - imagenet_ai_0508_adm
or standard custom class directories.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from .extractors import MultiBranchFeatureExtractor
from .resnet_baseline import ResNet18FeatureExtractor


class GenImageDatasetLoader:
    """Loader and Feature Cache Generator for the GenImage Dataset benchmark."""

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.multi_extractor = MultiBranchFeatureExtractor(device=device)
        self.resnet_extractor = ResNet18FeatureExtractor(device=device)

    def scan_dataset(self, dataset_dir: str, split: str = "val") -> Dict[str, List[str]]:
        """Scans dataset directory and returns a dictionary mapping class names to image paths."""
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        class_to_paths = {}

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        entries = sorted(os.listdir(dataset_dir))

        # Check if directory uses GenImage 'imagenet_ai_0508_*' format
        genimage_subfolders = [e for e in entries if "imagenet_ai_" in e and os.path.isdir(os.path.join(dataset_dir, e))]

        if genimage_subfolders:
            print(f"Detected official GenImage structure ({len(genimage_subfolders)} generator subsets)...")
            # Always add Real/Nature class
            class_to_paths["Real_ImageNet"] = []

            for folder in genimage_subfolders:
                gen_name = folder.replace("imagenet_ai_0508_", "").replace("imagenet_ai_", "")
                class_to_paths[gen_name] = []

                target_path = os.path.join(dataset_dir, folder, split)
                if not os.path.exists(target_path):
                    target_path = os.path.join(dataset_dir, folder)

                for root, _, files in os.walk(target_path):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in valid_exts:
                            full_p = os.path.join(root, file)
                            lower_root = root.lower()
                            if "nature" in lower_root or "real" in lower_root:
                                class_to_paths["Real_ImageNet"].append(full_p)
                            else:
                                class_to_paths[gen_name].append(full_p)
        else:
            # Standard class_name/ directory structure
            print("Detected standard class subfolder structure...")
            subdirs = [e for e in entries if os.path.isdir(os.path.join(dataset_dir, e))]

            for sdir in subdirs:
                class_to_paths[sdir] = []
                spath = os.path.join(dataset_dir, sdir)
                for root, _, files in os.walk(spath):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in valid_exts:
                            class_to_paths[sdir].append(os.path.join(root, file))

        # Filter empty classes
        class_to_paths = {k: v for k, v in class_to_paths.items() if len(v) > 0}
        return class_to_paths

    def extract_and_cache(
        self,
        dataset_dir: str,
        cache_file: str,
        split: str = "val",
        max_samples_per_class: Optional[int] = 500,
    ) -> Dict[str, Union[np.ndarray, List[str]]]:
        """Extracts features across all 6 model modes (ResNet-18, Baseline, Model A, B, C, All) and caches to disk."""
        if os.path.exists(cache_file):
            print(f"--> Loading cached GenImage benchmark features from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            return {
                "features_by_mode": data["features_by_mode"].item(),
                "labels": data["labels"],
                "class_names": list(data["class_names"]),
            }

        class_to_paths = self.scan_dataset(dataset_dir, split=split)
        class_names = sorted(list(class_to_paths.keys()))

        print(f"--> Found {len(class_names)} classes: {class_names}")

        modes = ["resnet18", "baseline", "model_a", "model_b", "model_c", "all"]
        features_by_mode = {m: [] for m in modes}
        labels = []

        total_processed = 0

        for class_idx, class_name in enumerate(class_names):
            paths = class_to_paths[class_name]
            if max_samples_per_class is not None:
                paths = paths[:max_samples_per_class]

            print(f"  • Extracting features for '{class_name}' ({len(paths)} images)...")

            for p in paths:
                try:
                    with Image.open(p) as img:
                        img_rgb = img.convert("RGB")

                        # 1. ResNet-18 RGB baseline feature
                        res_feat = self.resnet_extractor.extract(img_rgb)
                        features_by_mode["resnet18"].append(res_feat)

                        # 2. Proposed & Ablation Multi-Branch features
                        for mode in ["baseline", "model_a", "model_b", "model_c", "all"]:
                            mb_out = self.multi_extractor.extract_features(img_rgb, mode=mode)
                            features_by_mode[mode].append(mb_out["fused"])

                        labels.append(class_idx)
                        total_processed += 1

                        if total_processed % 100 == 0:
                            print(f"    Processed {total_processed} images...")
                except Exception as e:
                    print(f"    Warning: Error loading {p}: {e}")

        # Convert to numpy arrays
        features_dict = {m: np.array(features_by_mode[m], dtype=np.float32) for m in modes}
        labels_arr = np.array(labels, dtype=np.int64)

        # Cache to disk
        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(
            cache_file,
            features_by_mode=features_dict,
            labels=labels_arr,
            class_names=class_names,
        )
        print(f"--> Successfully cached {total_processed} samples to {cache_file}!")

        return {
            "features_by_mode": features_dict,
            "labels": labels_arr,
            "class_names": class_names,
        }
