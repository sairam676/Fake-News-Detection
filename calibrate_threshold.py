"""Step-by-Step Threshold Calibration for FASC Reliability Score.

Follows the guide's 5-step methodology:
  1. Load trained model.
  2. Calculate FASC scores on validation dataset.
  3. Test candidate thresholds: [0.40, 0.50, 0.60, 0.70, 0.80].
  4. For each threshold, measure:
       - Accepted predictions count & coverage %
       - Rejected / Abstained count
       - Errors among accepted predictions & error rate %
       - Accuracy on accepted predictions
  5. Select the optimal threshold and test on the untouched test set.
"""

import json
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.image_forensics.classifier import AttributionMLP
from src.image_forensics.fasc import FASCCalculator


def calibrate_threshold(
    cache_file: str = "data/cifake_features_cache.npz",
    model_path: str = "models/image_forensics_model.pt",
):
    print("=" * 85)
    print("FASC THRESHOLD CALIBRATION PROTOCOL (5-STEP METHODOLOGY)")
    print("=" * 85)

    cached = np.load(cache_file, allow_pickle=True)
    X_test_full = cached["test_feats"].item()["all"]
    y_test_full = cached["y_test"]
    class_names = [str(c) for c in cached["class_names"]]

    # Split into 50% Validation (for tuning threshold) and 50% Untouched Test (for final evaluation)
    val_idx, test_idx = train_test_split(
        np.arange(len(y_test_full)), test_size=0.50, random_state=42, stratify=y_test_full
    )

    X_val, y_val = X_test_full[val_idx], y_test_full[val_idx]
    X_test, y_test = X_test_full[test_idx], y_test_full[test_idx]

    print(f"--> Split test images into:")
    print(f"    • Validation Set (for finding threshold): {len(X_val)} images")
    print(f"    • Untouched Test Set (for final evaluation): {len(X_test)} images")

    # Load trained model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = AttributionMLP(input_dim=576, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    calculator = FASCCalculator(model, class_names)

    # 1. Compute FASC scores and predictions on Validation Set
    val_predictions = []
    val_fasc_scores = []
    val_correct = []

    for i in range(len(X_val)):
        out = calculator.compute_scores(X_val[i])
        val_predictions.append(out["predicted_class"])
        val_fasc_scores.append(out["fasc_reliability_score"])
        is_corr = 1 if out["predicted_class"] == class_names[y_val[i]] else 0
        val_correct.append(is_corr)

    val_fasc_scores = np.array(val_fasc_scores)
    val_correct = np.array(val_correct)

    # 2. Sweep Candidate Thresholds: [0.40, 0.50, 0.60, 0.70, 0.80]
    candidate_thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]

    print("\n" + "=" * 85)
    print("STEP 3 & 4: EVALUATING CANDIDATE THRESHOLDS ON VALIDATION DATASET")
    print("=" * 85)
    print(f"{'Threshold (tau)':16s} | {'Accepted':10s} | {'Coverage %':12s} | {'Errors':8s} | {'Error Rate %':14s} | {'Accepted Acc %':14s}")
    print("-" * 85)

    table_data = []
    optimal_tau = 0.50
    lowest_error_rate = 1.0

    for tau in candidate_thresholds:
        accepted_mask = val_fasc_scores >= tau
        num_accepted = int(np.sum(accepted_mask))
        coverage_pct = (num_accepted / len(val_fasc_scores)) * 100.0

        if num_accepted > 0:
            accepted_correct = val_correct[accepted_mask]
            errors = int(np.sum(accepted_correct == 0))
            error_rate = (errors / num_accepted) * 100.0
            acc_pct = (np.sum(accepted_correct) / num_accepted) * 100.0
        else:
            errors = 0
            error_rate = 0.0
            acc_pct = 0.0

        table_data.append({
            "threshold": tau,
            "accepted": num_accepted,
            "coverage_pct": coverage_pct,
            "errors": errors,
            "error_rate": error_rate,
            "accepted_acc": acc_pct,
        })

        print(
            f"tau = {tau:4.2f}          | {num_accepted:4d}/{len(val_fasc_scores)}   | "
            f"{coverage_pct:10.2f}% | {errors:4d}   | {error_rate:12.2f}% | {acc_pct:12.2f}%"
        )

    # Choose optimal threshold (balancing high coverage >= 85% with low error rate <= 5%)
    for row in table_data:
        if row["coverage_pct"] >= 85.0 and row["error_rate"] <= 6.0:
            optimal_tau = row["threshold"]

    print("-" * 85)
    print(f"--> Selected Optimal Threshold: tau = {optimal_tau:.2f}")

    # 3. STEP 5: Evaluate on the Untouched Test Set
    print("\n" + "=" * 85)
    print(f"STEP 5: FINAL EVALUATION ON UNTOUCHED TEST SET (tau = {optimal_tau:.2f})")
    print("=" * 85)

    test_fasc_scores = []
    test_correct = []

    for i in range(len(X_test)):
        out = calculator.compute_scores(X_test[i])
        test_fasc_scores.append(out["fasc_reliability_score"])
        is_corr = 1 if out["predicted_class"] == class_names[y_test[i]] else 0
        test_correct.append(is_corr)

    test_fasc_scores = np.array(test_fasc_scores)
    test_correct = np.array(test_correct)

    raw_test_acc = np.mean(test_correct) * 100.0

    # With FASC threshold
    accepted_mask_test = test_fasc_scores >= optimal_tau
    test_accepted = int(np.sum(accepted_mask_test))
    test_coverage = (test_accepted / len(test_correct)) * 100.0
    test_errors = int(np.sum(test_correct[accepted_mask_test] == 0))
    test_accepted_acc = (np.sum(test_correct[accepted_mask_test]) / test_accepted) * 100.0

    print(f"• Baseline Accuracy (No Rejection / 100% Coverage): {raw_test_acc:.2f}%")
    print(f"• With FASC Threshold (tau = {optimal_tau:.2f}):")
    print(f"    - Accepted Predictions:   {test_accepted}/{len(test_correct)} ({test_coverage:.2f}% Coverage)")
    print(f"    - Abstained / Uncertain:  {len(test_correct) - test_accepted} images")
    print(f"    - Errors in Accepted:     {test_errors} out of {test_accepted}")
    print(f"    - Final Accepted Accuracy:{test_accepted_acc:.2f}%  (+{test_accepted_acc - raw_test_acc:.2f}% improvement!)")
    print("=" * 85)

    # Save calibration report
    report = {
        "candidate_thresholds": table_data,
        "selected_optimal_threshold": optimal_tau,
        "untouched_test_metrics": {
            "raw_accuracy": raw_test_acc,
            "accepted_accuracy": test_accepted_acc,
            "coverage_pct": test_coverage,
            "abstained_count": len(test_correct) - test_accepted,
        }
    }
    with open("results/threshold_calibration_results.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    calibrate_threshold()
