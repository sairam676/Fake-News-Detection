"""Save the final trained 576-d Multi-Branch model to disk for production inference."""

import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.image_forensics.classifier import AttributionMLP
from src.image_forensics.trainer import ForensicsTrainer, ForensicsFeatureDataset

def save_trained_image_model(cache_file="data/cifake_features_cache.npz", save_path="models/image_forensics_model.pt"):
    print("=" * 70)
    print("SAVING FINAL 576-d MULTI-BRANCH IMAGE FORENSICS MODEL")
    print("=" * 70)

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Feature cache {cache_file} not found. Run train_cifake_experiments.py first.")

    cached = np.load(cache_file, allow_pickle=True)
    X_train = cached["train_feats"].item()["all"]
    y_train = cached["y_train"]
    X_test = cached["test_feats"].item()["all"]
    y_test = cached["y_test"]
    class_names = list(cached["class_names"])

    print(f"--> Loaded pre-extracted features: {len(X_train)} train, {len(X_test)} test")
    print(f"--> Target classes: {class_names}")

    train_ds = ForensicsFeatureDataset(X_train, y_train)
    test_ds = ForensicsFeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Training final 576-d model on {device.upper()}...")

    trainer = ForensicsTrainer(
        num_classes=len(class_names),
        input_dim=576,
        hidden_dim=256,
        device=device,
    )

    metrics = trainer.train_full(train_loader, test_loader, epochs=15, verbose=True)
    print(f"\n--> Final Model Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    # Save model weights and class mapping
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "state_dict": trainer.model.state_dict(),
        "input_dim": 576,
        "num_classes": len(class_names),
        "class_names": class_names,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"]
        }
    }, save_path)

    classes_path = os.path.join(os.path.dirname(save_path), "image_classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"--> Successfully saved model weights to: {save_path}")
    print(f"--> Saved class mapping to: {classes_path}")
    print("=" * 70)

if __name__ == "__main__":
    save_trained_image_model()
