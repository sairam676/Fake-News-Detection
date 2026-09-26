"""Dataset loader and feature caching for GenImage and synthetic forensics benchmarks."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from .extractors import MultiBranchFeatureExtractor


class ForensicsFeatureDataset(Dataset):
    """PyTorch Dataset holding pre-extracted features and class labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ForensicsDatasetManager:
    """Manages feature extraction, caching, and dataset splitting for generator attribution."""

    def __init__(self, extractor: Optional[MultiBranchFeatureExtractor] = None):
        if extractor is None:
            self.extractor = MultiBranchFeatureExtractor()
        else:
            self.extractor = extractor

    def extract_and_cache(
        self,
        dataset_dir: str,
        cache_file: str,
        mode: str = "all",
        max_samples_per_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Scans dataset directory, extracts multi-branch features, and caches to .npz file."""
        if os.path.exists(cache_file):
            print(f"--> Loading pre-cached features from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            return data["features"], data["labels"], list(data["class_names"])

        print(f"--> Scanning dataset directory: {dataset_dir} (mode={mode})...")
        class_names = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])

        if not class_names:
            raise ValueError(f"No subdirectories (classes) found in {dataset_dir}")

        features_list = []
        labels_list = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(dataset_dir, class_name)
            image_files = [
                f for f in os.listdir(class_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            if max_samples_per_class is not None:
                image_files = image_files[:max_samples_per_class]

            print(f"  • Processing class '{class_name}' ({len(image_files)} images)...")
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        feat_dict = self.extractor.extract_features(img, mode=mode)
                        features_list.append(feat_dict["fused"])
                        labels_list.append(class_idx)
                except Exception as e:
                    print(f"    Warning: Could not process {img_path}: {e}")

        features = np.array(features_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(cache_file, features=features, labels=labels, class_names=class_names)
        print(f"--> Extracted & cached {len(features)} samples to {cache_file}")

        return features, labels, class_names

    @staticmethod
    def generate_synthetic_dataset(
        num_classes: int = 5,
        samples_per_class: int = 150,
        mode: str = "all",
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generates synthetic feature data matching real feature distributions for fast unit testing."""
        np.random.seed(seed)
        dims_map = {"baseline": 384, "model_a": 64, "model_b": 448, "model_c": 512, "all": 576}
        dim = dims_map[mode]

        class_names = ["Real_ImageNet", "Midjourney_v5", "StableDiffusion_v1_5", "BigGAN", "Wukong"][:num_classes]

        features_list = []
        labels_list = []

        # Create distinct orthogonal basis directions for each generator class
        for c_idx in range(num_classes):
            center = np.zeros(dim)
            # Assign distinct feature dimension peaks per class
            stride = max(dim // num_classes, 1)
            center[c_idx * stride : (c_idx + 1) * stride] = 3.0

            noise = np.random.randn(samples_per_class, dim) * 0.3
            class_feats = center + noise
            class_feats = class_feats / np.linalg.norm(class_feats, axis=1, keepdims=True)
            features_list.append(class_feats)
            labels_list.append(np.full(samples_per_class, c_idx))

        features = np.vstack(features_list).astype(np.float32)
        labels = np.concatenate(labels_list).astype(np.int64)

        return features, labels, class_names
