"""Verification script for Step 1: Multi-Branch Feature Extractors."""

import numpy as np
from PIL import Image
from src.image_forensics.extractors import MultiBranchFeatureExtractor

def test_extractors():
    print("=" * 65)
    print("STEP 1: Testing Multi-Branch Feature Extraction Pipeline")
    print("=" * 65)

    # 1. Create a dummy synthetic image (224x224 RGB)
    print("[1/3] Generating synthetic test image (224x224 RGB)...")
    np.random.seed(42)
    sample_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    sample_img = Image.fromarray(sample_arr)

    # 2. Initialize the multi-branch extractor
    print("[2/3] Initializing MultiBranchFeatureExtractor (DINOv2-small ViT-S/14)...")
    extractor = MultiBranchFeatureExtractor(dinov2_model="dinov2_vits14")

    # 3. Test each ablation mode
    modes = ["baseline", "model_a", "model_b", "model_c", "all"]
    expected_dims = {
        "baseline": 384,     # DINOv2 CLS (Spatial/Semantics)
        "model_a": 64,       # Low-Bit Fingerprint only (Forensic)
        "model_b": 448,      # DINOv2 (384) + Style (64)
        "model_c": 512,      # DINOv2 (384) + Style (64) + Frequency (64)
        "all": 576,          # DINOv2 (384) + Style (64) + Frequency (64) + Fingerprint (64)
    }

    print("\n[3/3] Extracting features across all ablation modes:")
    for mode in modes:
        out = extractor.extract_features(sample_img, mode=mode)
        feat = out["fused"]
        expected = expected_dims[mode]
        status = "PASSED" if feat.shape[0] == expected else "FAILED"
        print(f"  • Mode [{mode:9s}]: shape={feat.shape} | expected={expected} | {status}")
        assert feat.shape[0] == expected, f"Dimension mismatch for {mode}: got {feat.shape[0]}, expected {expected}"

    print("\nComponent Breakdown (from 'all' mode, total 576-d):")
    all_out = extractor.extract_features(sample_img, mode="all")
    print(f"  1. Spatial (DINOv2 CLS):         {all_out['spatial'].shape} (high-level visual semantics)")
    print(f"  2. Style / Mid-Level:            {all_out['style'].shape} (patch correlation / Gram moments)")
    print(f"  3. Spectral (2D-FFT):            {all_out['frequency'].shape} (radial grid frequency profile)")
    print(f"  4. Forensic Residual (LIDA):     {all_out['fingerprint'].shape} (low-bit plane noise statistics)")
    print(f"  --> Total Unified Fused Vector:  {all_out['fused'].shape}")

    print("\n" + "=" * 65)
    print("SUCCESS: Step 1 Feature Extraction module is verified and working!")
    print("=" * 65)

if __name__ == "__main__":
    test_extractors()
