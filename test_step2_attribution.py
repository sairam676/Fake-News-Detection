"""Verification script for Step 2: Generator Attribution Classifier & Open-Set Unknown Detection."""

import numpy as np
from src.image_forensics.dataset import ForensicsDatasetManager
from src.image_forensics.classifier import OpenSetAttributor
from src.image_forensics.trainer import run_ablation_benchmark


def test_step2_attribution():
    print("=" * 70)
    print("STEP 2: Generator Attribution Classifier & Open-Set 'Unknown' Benchmark")
    print("=" * 70)

    # 1. Setup Synthetic Known Classes
    known_classes = ["Real_ImageNet", "Midjourney_v5", "StableDiffusion_v1_5", "BigGAN"]
    num_known = len(known_classes)

    print(f"[1/4] Preparing dataset split for {num_known} known generator classes...")
    print(f"      Known Classes: {known_classes}")

    # Generate synthetic features for training and validation across modes
    train_feats_by_mode = {}
    val_feats_by_mode = {}

    modes = ["baseline", "model_a", "model_b", "model_c", "all"]

    for mode in modes:
        X_train, y_train, _ = ForensicsDatasetManager.generate_synthetic_dataset(
            num_classes=num_known, samples_per_class=200, mode=mode, seed=42
        )
        X_val, y_val, _ = ForensicsDatasetManager.generate_synthetic_dataset(
            num_classes=num_known, samples_per_class=50, mode=mode, seed=123
        )
        train_feats_by_mode[mode] = X_train
        val_feats_by_mode[mode] = X_val

    # 2. Run Comparative Ablation Benchmark across all 5 models
    print("\n[2/4] Running Comparative Training Benchmark for 5 Feature Modes...")
    benchmark_results = run_ablation_benchmark(
        train_features_by_mode=train_feats_by_mode,
        val_features_by_mode=val_feats_by_mode,
        train_labels=y_train,
        val_labels=y_val,
        class_names=known_classes,
        epochs=15,
        batch_size=32,
    )

    # Print Ablation Results Table
    print("\n" + "=" * 70)
    print("STEP 2: COMPARATIVE ABLATION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model Name':15s} | {'Input Dim':9s} | {'Accuracy':10s} | {'Macro F1':10s} | {'Precision':10s} | {'Recall':10s}")
    print("-" * 75)

    for mode in modes:
        res = benchmark_results[mode]
        print(
            f"{mode:15s} | {res['input_dim']:9d} | "
            f"{res['accuracy']*100:9.2f}% | {res['macro_f1']:10.4f} | "
            f"{res['precision']:10.4f} | {res['recall']:10.4f}"
        )
    print("-" * 75)

    # 3. Test Open-Set / Unknown Generator Detection
    print("\n[3/4] Testing Open-Set Detection on Unseen Generator ('Flux_v1_Unseen')...")

    best_model = benchmark_results["all"]["trained_model"]
    attributor = OpenSetAttributor(
        model=best_model,
        class_names=known_classes,
        confidence_threshold=0.50,
    )

    # Generate an out-of-distribution / unseen generator sample (uniform noise across all features, far from distinct class centers)
    np.random.seed(999)
    unseen_sample = np.random.randn(576).astype(np.float32)
    unseen_sample = unseen_sample / np.linalg.norm(unseen_sample)

    prediction = attributor.predict(unseen_sample)

    print("\nOpen-Set Prediction Output for Unseen Generator:")
    print(f"  • Raw Internal Best Class:    {prediction['raw_prediction']}")
    print(f"  • Model Confidence Score:     {prediction['confidence']*100:.2f}%")
    print(f"  • Confidence Threshold:       {attributor.confidence_threshold * 100:.2f}%")
    print(f"  • Final Output Prediction:   '{prediction['predicted_label']}'")
    print(f"  • Is Identified as Known?:    {prediction['is_known_generator']}")

    assert not prediction["is_known_generator"], "Error: Unseen generator was incorrectly matched to known class!"
    assert prediction["predicted_label"] == "Unknown / Unidentified Generator", "Error: Open-set label mismatch!"

    print("\n" + "=" * 70)
    print("SUCCESS: Step 2 Attribution Training & Open-Set Unknown Detection Verified!")
    print("=" * 70)


if __name__ == "__main__":
    test_step2_attribution()
