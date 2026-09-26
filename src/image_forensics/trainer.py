"""Trainer and Evaluator for Generator Attribution & Ablation Experiments."""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .classifier import AttributionMLP, OpenSetAttributor
from .dataset import ForensicsFeatureDataset


class ForensicsTrainer:
    """Trains and evaluates AttributionMLP models across feature ablation modes."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.model = AttributionMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for feats, labels in dataloader:
            feats, labels = feats.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(labels)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []

        for feats, labels in dataloader:
            feats = feats.to(self.device)
            logits = self.model(feats)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
        }

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        verbose: bool = False,
    ) -> Dict[str, float]:
        best_acc = 0.0
        best_metrics = {}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            metrics["loss"] = loss

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_metrics = metrics

            if verbose and (epoch % 5 == 0 or epoch == epochs):
                print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {loss:.4f} | Val Acc: {metrics['accuracy']*100:.2f}% | Val F1: {metrics['macro_f1']:.4f}")

        return best_metrics


def run_ablation_benchmark(
    train_features_by_mode: Dict[str, np.ndarray],
    val_features_by_mode: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    class_names: List[str],
    epochs: int = 15,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Runs training across all 5 feature ablation modes and compiles a comparative benchmark table."""
    results = {}
    num_classes = len(class_names)

    modes = ["baseline", "model_a", "model_b", "model_c", "all"]

    for mode in modes:
        X_train = train_features_by_mode[mode]
        X_val = val_features_by_mode[mode]

        train_ds = ForensicsFeatureDataset(X_train, train_labels)
        val_ds = ForensicsFeatureDataset(X_val, val_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        input_dim = X_train.shape[1]
        print(f"\n--> Training Mode [{mode:9s}] (Input Dim: {input_dim:3d})...")

        trainer = ForensicsTrainer(
            num_classes=num_classes,
            input_dim=input_dim,
            hidden_dim=256,
            device=device,
        )

        metrics = trainer.train_full(train_loader, val_loader, epochs=epochs, verbose=True)
        results[mode] = metrics
        results[mode]["input_dim"] = input_dim
        results[mode]["trained_model"] = trainer.model

    return results
