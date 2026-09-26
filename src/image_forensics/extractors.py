"""Multi-Branch Feature Extractors for AI-Generated Image Attribution and Forensics.

Branches:
  1. DINOv2 Spatial / Semantic Extractor: Global semantic and visual composition (384-d).
  2. Style / Mid-Level Representation Extractor: Texture and channel co-occurrence / Gram moments (64-d).
  3. Frequency / Spectral Extractor: 2D-FFT azimuthal/radial frequency profile (64-d).
  4. Forensic Residual Extractor (LIDA): Low-bit plane fingerprint noise statistics (64-d).
"""

from typing import Dict, Optional, Union
import numpy as np
from PIL import Image
import torch


def preprocess_image_tensor(image: Union[Image.Image, np.ndarray], target_size: int = 224, device: str = "cpu") -> torch.Tensor:
    """Preprocesses PIL or NumPy image into normalized PyTorch tensor for vision backbones."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to target size (224x224)
    image_resized = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    arr = np.array(image_resized, dtype=np.float32) / 255.0  # [H, W, 3]

    # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (arr - mean) / std

    # [H, W, C] -> [1, C, H, W]
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


class DINOv2SpatialExtractor:
    """Extracts spatial and high-level semantic representations using DINOv2."""

    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        # Load directly via torch hub (facebookresearch/dinov2)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: Union[Image.Image, np.ndarray], return_patches: bool = False) -> Dict[str, np.ndarray]:
        """Extracts DINOv2 CLS token (384-d for small) and optionally patch tokens."""
        tensor = preprocess_image_tensor(image, target_size=224, device=self.device)

        if return_patches:
            feats = self.model.forward_features(tensor)
            cls_token = feats["x_norm_clstoken"].squeeze(0).cpu().numpy()
            patch_tokens = feats["x_norm_patchtokens"].squeeze(0).cpu().numpy()
            return {"cls_token": cls_token, "patch_tokens": patch_tokens}
        else:
            cls_token = self.model(tensor).squeeze(0).cpu().numpy()
            return {"cls_token": cls_token}


class FFTFrequencyExtractor:
    """Extracts 2D-FFT radial/spectral features to detect frequency artifacts."""

    def __init__(self, num_bins: int = 64):
        self.num_bins = num_bins

    def extract(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Computes 1D radial power spectrum profile from 2D-FFT."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"), dtype=np.float32)
        else:
            image = image.astype(np.float32)

        # Convert to luminance Y: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        h, w = gray.shape

        # 2D Fast Fourier Transform and shift DC component to center
        fft_2d = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_2d)
        magnitude_spectrum = np.abs(fft_shifted)
        log_spectrum = np.log1p(magnitude_spectrum)

        # Radial distance grid from center
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        max_radius = min(cx, cy)
        bin_edges = np.linspace(0, max_radius, self.num_bins + 1)

        radial_profile = np.zeros(self.num_bins, dtype=np.float32)
        for i in range(self.num_bins):
            mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = np.mean(log_spectrum[mask])
            else:
                radial_profile[i] = 0.0

        # L2 normalize
        norm = np.linalg.norm(radial_profile)
        if norm > 1e-8:
            radial_profile /= norm

        return radial_profile


class LowBitFingerprintExtractor:
    """Extracts low-bit plane forensic residual features based on LIDA (CVPR 2024)."""

    def __init__(self, num_features: int = 64):
        self.num_features = num_features

    def compute_low_bit_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Implements LIDA Eq. (2) & (3): Extracts 3 least significant bit planes."""
        image_uint8 = image.astype(np.uint8)
        # 3 LSBs mask is 0b00000111 = 7
        lsb_combined = image_uint8 & 7
        # Thresholding strategy: non-zero mapped to 255
        fingerprint = (lsb_combined > 0).astype(np.float32) * 255.0
        return fingerprint

    def extract(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extracts statistical noise characteristics of low-bit planes."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)

        fingerprint = self.compute_low_bit_fingerprint(image)

        # 1. Per-channel density of non-zero bits (3 values)
        channel_means = np.mean(fingerprint > 0, axis=(0, 1))

        # 2. Local spatial noise transitions (first-order horizontal & vertical differences)
        diff_h = np.abs(np.diff(fingerprint, axis=1))
        diff_v = np.abs(np.diff(fingerprint, axis=0))

        h_means = np.mean(diff_h, axis=(0, 1))
        h_stds = np.std(diff_h, axis=(0, 1))
        v_means = np.mean(diff_v, axis=(0, 1))
        v_stds = np.std(diff_v, axis=(0, 1))

        # 3. Block-level variance distribution across 8x8 patches
        h, w, c = fingerprint.shape
        bh, bw = max(h // 8, 1), max(w // 8, 1)
        grid_features = []
        for i in range(8):
            for j in range(8):
                patch = fingerprint[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw, :]
                grid_features.append(np.mean(patch > 0))

        stats = np.concatenate([
            channel_means,
            h_means,
            h_stds,
            v_means,
            v_stds,
            np.array(grid_features[: 64 - 15], dtype=np.float32),
        ])

        if len(stats) < self.num_features:
            stats = np.pad(stats, (0, self.num_features - len(stats)))
        else:
            stats = stats[: self.num_features]

        norm = np.linalg.norm(stats)
        if norm > 1e-8:
            stats /= norm

        return stats.astype(np.float32)


class StyleMidLevelExtractor:
    """Extracts style and mid-level structural features via Gram/covariance statistics."""

    def __init__(self, num_features: int = 64):
        self.num_features = num_features

    def extract(self, image: Union[Image.Image, np.ndarray], dinov2_patches: Optional[np.ndarray] = None) -> np.ndarray:
        """Extracts texture and style correlation features."""
        if dinov2_patches is not None:
            # Using DINOv2 patch token cross-correlations
            # dinov2_patches shape: [256, 384]
            patches = dinov2_patches - np.mean(dinov2_patches, axis=0, keepdims=True)
            cov = np.dot(patches.T, patches) / max(patches.shape[0] - 1, 1)
            eigenvals = np.linalg.eigvalsh(cov)
            eigenvals = np.sort(eigenvals)[::-1]
            feature = eigenvals[: self.num_features]
        else:
            if isinstance(image, Image.Image):
                img_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            else:
                img_arr = image.astype(np.float32) / 255.0

            # Gradient channels
            gx = np.gradient(img_arr, axis=1)
            gy = np.gradient(img_arr, axis=0)

            channels = np.concatenate([img_arr, gx, gy], axis=-1)  # [H, W, 9]
            h, w, c = channels.shape
            flat_channels = channels.reshape(-1, c)

            # Gram matrix G = (C^T * C) / (H * W) -> [9, 9]
            gram = np.dot(flat_channels.T, flat_channels) / (h * w)
            gram_triu = gram[np.triu_indices(c)]

            patch_vars = [
                np.var(img_arr[: h // 2, : w // 2]),
                np.var(img_arr[: h // 2, w // 2 :]),
                np.var(img_arr[h // 2 :, : w // 2]),
                np.var(img_arr[h // 2 :, w // 2 :]),
            ]

            feature = np.concatenate([gram_triu, patch_vars])

        if len(feature) < self.num_features:
            feature = np.pad(feature, (0, self.num_features - len(feature)))
        else:
            feature = feature[: self.num_features]

        norm = np.linalg.norm(feature)
        if norm > 1e-8:
            feature /= norm

        return feature.astype(np.float32)


class MultiBranchFeatureExtractor:
    """Unified multi-branch extractor supporting all ablation configurations."""

    def __init__(self, dinov2_model: str = "dinov2_vits14", device: Optional[str] = None):
        self.spatial_extractor = DINOv2SpatialExtractor(model_name=dinov2_model, device=device)
        self.freq_extractor = FFTFrequencyExtractor(num_bins=64)
        self.fingerprint_extractor = LowBitFingerprintExtractor(num_features=64)
        self.style_extractor = StyleMidLevelExtractor(num_features=64)

    def extract_features(
        self, image: Union[Image.Image, np.ndarray], mode: str = "all"
    ) -> Dict[str, np.ndarray]:
        """Extracts features based on the chosen ablation mode.

        Modes:
          - 'baseline': DINOv2 CLS token (384-d)
          - 'model_a': Low-bit fingerprint only (64-d)
          - 'model_b': DINOv2 + Style (448-d)
          - 'model_c': DINOv2 + Style + Frequency (512-d)
          - 'all': DINOv2 + Style + Frequency + Fingerprint (576-d)
        """
        # 1. Spatial & DINOv2 patch representation
        dinov2_out = self.spatial_extractor.extract(image, return_patches=True)
        feat_spatial = dinov2_out["cls_token"]
        dinov2_patches = dinov2_out["patch_tokens"]

        # 2. Style / Mid-level representation
        feat_style = self.style_extractor.extract(image, dinov2_patches=dinov2_patches)

        # 3. Frequency profile
        feat_freq = self.freq_extractor.extract(image)

        # 4. Low-bit forensic fingerprint
        feat_fingerprint = self.fingerprint_extractor.extract(image)

        # Construct fused feature vector according to requested mode
        if mode == "baseline":
            fused = feat_spatial
        elif mode == "model_a":
            fused = feat_fingerprint
        elif mode == "model_b":
            fused = np.concatenate([feat_spatial, feat_style])
        elif mode == "model_c":
            fused = np.concatenate([feat_spatial, feat_style, feat_freq])
        elif mode == "all":
            fused = np.concatenate([feat_spatial, feat_style, feat_freq, feat_fingerprint])
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

        return {
            "fused": fused.astype(np.float32),
            "spatial": feat_spatial.astype(np.float32),
            "style": feat_style.astype(np.float32),
            "frequency": feat_freq.astype(np.float32),
            "fingerprint": feat_fingerprint.astype(np.float32),
            "dim": len(fused),
        }
