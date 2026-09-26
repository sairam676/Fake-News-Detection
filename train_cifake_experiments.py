"""CIFAKE Dataset Benchmark for Lightweight Research on Laptops/Desktops.

Uses the CIFAKE dataset (Bird & Lotfi 2024, WACV/IEEE - 120,000 images):
  - Label 0: Real photos (CIFAR-10)
  - Label 1: AI-generated synthetic images (Latent Diffusion / Stable Diffusion)
"""

import argparse
import json
import os
from typing import Dict, List
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from src.image_forensics.classifier import AttributionMLP, OpenSetAttributor
from src.image_forensics.extractors import MultiBranchFeatureExtractor
from src.image_forensics.resnet_baseline import ResNet18FeatureExtractor
from src.image_forensics.trainer import ForensicsTrainer, ForensicsFeatureDataset
from torch.utils.data import DataLoader


def run_cifake_experiments(
    num_train: int = 1000,
    num_test: int = 400,
    cache_file: str = "data/cifake_features_cache.npz",
    epochs: int = 15,
    batch_size: int = 32,
):
    print("=" * 80)
    print("CIFAKE BENCHMARK: RESNET-18 & PROPOSED MULTI-BRANCH EXPERIMENTS")
    print("Dataset: CIFAKE (Bird & Lotfi 2024, WACV/IEEE - 120,000 Real & Diffusion Images)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Using Computation Device: {device.upper()}")

    # 1. Load and Cache Features if not already cached
    modes = ["resnet18", "baseline", "model_a", "model_b", "model_c", "all"]

    if os.path.exists(cache_file):
        print(f"--> Loading pre-cached CIFAKE features from {cache_file}...")
        cached = np.load(cache_file, allow_pickle=True)
        train_feats_by_mode = cached["train_feats"].item()
        test_feats_by_mode = cached["test_feats"].item()
        y_train = cached["y_train"]
        y_test = cached["y_test"]
        class_names = list(cached["class_names"])
    else:
        print(f"[1/4] Loading CIFAKE dataset from Hugging Face ({num_train} train, {num_test} test)...")
        dataset_hf = load_dataset("dragonintelligence/CIFAKE-image-dataset")

        train_split = dataset_hf["train"].shuffle(seed=42).select(range(num_train))
        test_split = dataset_hf["test"].shuffle(seed=123).select(range(num_test))

        class_names = ["Real_Photo", "AI_Diffusion_Synthetic"]

        multi_extractor = MultiBranchFeatureExtractor(device=device)
        resnet_extractor = ResNet18FeatureExtractor(device=device)

        def extract_split_features(split_data, split_name: str):
            print(f"  • Extracting features for {split_name} split ({len(split_data)} images)...")
            feats_dict = {m: [] for m in modes}
            labels = []

            for idx, item in enumerate(split_data):
                img = item["image"].convert("RGB")
                label = item["label"]
                labels.append(label)

                # ResNet-18 baseline (512-d)
                res_f = resnet_extractor.extract(img)
                feats_dict["resnet18"].append(res_f)

                # Proposed & Ablation Multi-Branch features
                for m in ["baseline", "model_a", "model_b", "model_c", "all"]:
                    mb_out = multi_extractor.extract_features(img, mode=m)
                    feats_dict[m].append(mb_out["fused"])

                if (idx + 1) % 250 == 0 or (idx + 1) == len(split_data):
                    print(f"    Extracted {idx + 1}/{len(split_data)} images...")

            feats_arr = {m: np.array(feats_dict[m], dtype=np.float32) for m in modes}
            return feats_arr, np.array(labels, dtype=np.int64)

        train_feats_by_mode, y_train = extract_split_features(train_split, "Train")
        test_feats_by_mode, y_test = extract_split_features(test_split, "Test")

        os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
        np.savez(
            cache_file,
            train_feats=train_feats_by_mode,
            test_feats=test_feats_by_mode,
            y_train=y_train,
            y_test=y_test,
            class_names=class_names,
        )
        print(f"--> Extracted and cached features to '{cache_file}'!")

    print(f"--> Split Info: Train Samples = {len(y_train)} | Test Samples = {len(y_test)}")
    print(f"--> Classes: {class_names}")

    # 2. Train all 6 Model Variations
    print("\n[2/4] Training all 6 Model Variations on CIFAKE...")

    results = {}

    for mode in modes:
        X_tr = train_feats_by_mode[mode]
        X_te = test_feats_by_mode[mode]

        tr_ds = ForensicsFeatureDataset(X_tr, y_train)
        te_ds = ForensicsFeatureDataset(X_te, y_test)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

        input_dim = X_tr.shape[1]
        print(f"\n  • Training Model [{mode:9s}] (Input Dimension: {input_dim:3d})...")

        trainer = ForensicsTrainer(
            num_classes=len(class_names),
            input_dim=input_dim,
            hidden_dim=256,
            device=device,
        )

        metrics = trainer.train_full(tr_loader, te_loader, epochs=epochs, verbose=False)
        results[mode] = {
            "input_dim": input_dim,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "loss": metrics["loss"],
        }
        results[mode]["model_obj"] = trainer.model

    # 3. Print Comparative Performance Table
    print("\n" + "=" * 85)
    print("CIFAKE BENCHMARK: COMPARATIVE PERFORMANCE TABLE (WACV/IEEE Benchmark)")
    print("=" * 85)
    print(f"{'Model Name':18s} | {'Input Feature / Domain':28s} | {'Dim':5s} | {'Accuracy':9s} | {'Macro F1':9s}")
    print("-" * 85)

    model_descriptions = {
        "resnet18": "ResNet-18 RGB Backbone (Baseline)",
        "baseline": "DINOv2 ViT-S CLS Token (Baseline)",
        "model_a": "Low-Bit Forensic Residual (LIDA)",
        "model_b": "DINOv2 + Style Gram Matrix",
        "model_c": "DINOv2 + Style + 2D-FFT Spectral",
        "all": "Full Fused Multi-Branch (Proposed)",
    }

    for mode in modes:
        res = results[mode]
        desc = model_descriptions[mode]
        print(
            f"{mode:18s} | {desc:28s} | {res['input_dim']:5d} | "
            f"{res['accuracy']*100:8.2f}% | {res['macro_f1']:9.4f}"
        )
    print("-" * 85)

    # 4. Open-Set Unknown Detection Evaluation
    print("\n[3/4] Testing Open-Set Detection for Unseen Out-of-Distribution Image...")
    best_model = results["all"]["model_obj"]
    attributor = OpenSetAttributor(
        model=best_model,
        class_names=class_names,
        confidence_threshold=0.50,
        device=device,
    )

    np.random.seed(999)
    unseen_vec = np.random.randn(576).astype(np.float32)
    unseen_vec = unseen_vec / np.linalg.norm(unseen_vec)

    open_set_res = attributor.predict(unseen_vec)

    print("  • Output Prediction:    ", f"'{open_set_res['predicted_label']}'")
    print("  • Confidence Score:     ", f"{open_set_res['confidence']*100:.2f}%")
    print("  • Is Known Class?:      ", open_set_res["is_known_generator"])

    # Export JSON results
    json_results = {
        m: {k: v for k, v in results[m].items() if k != "model_obj"} for m in results
    }
    json_results["open_set_test"] = open_set_res

    os.makedirs("results", exist_ok=True)
    with open("results/cifake_benchmark_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print("\n--> Benchmark results saved to 'results/cifake_benchmark_results.json'")
    print("=" * 80)
    print("CIFAKE EXPERIMENT BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAKE Multi-Model Forensic Benchmark")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num_test", type=int, default=400, help="Number of testing samples")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_cifake_experiments(
        num_train=args.num_train,
        num_test=args.num_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
