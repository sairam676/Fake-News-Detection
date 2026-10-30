"""Feature-Ablation Sensitivity-Based Confidence (FASC) / Forensic Evidence Reliability Score (FERS).

Implements:
  1. Base Confidence: C(x) = p_all(y_hat | x)
  2. Branch Ablation Sensitivity:
       D(x) = 1/4 * sum_{k=1..4} | p_all(y_hat | x) - p_{-k}(y_hat | x) |
     where p_{-k} is the prediction when branch k is ablated (zero-masked).
  3. FASC Reliability Score:
       FASC(x) = C(x) * [1 - D(x)]
  4. Baselines for comparative research:
       - Maximum Softmax Probability (MSP)
       - Top-Two Probability Margin
       - Feature Branch Agreement
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from .classifier import AttributionMLP


# Slices corresponding to each of the 4 forensic branches in the 576-d vector
BRANCH_SLICES = {
    "dinov2_spatial": (0, 384),        # 384 dims
    "style_gram":     (384, 448),      # 64 dims
    "fft_frequency":  (448, 512),      # 64 dims
    "lowbit_residual": (512, 576),     # 64 dims
}


class FASCCalculator:
    """Computes Feature-Ablation Sensitivity and Reliability Scores."""

    def __init__(
        self,
        model: AttributionMLP,
        class_names: List[str],
        reliability_threshold: float = 0.50,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.reliability_threshold = reliability_threshold
        self.device = device

    @torch.no_grad()
    def compute_scores(self, feature_vector: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[float, str, bool, Dict]]:
        """Computes Confidence, Sensitivity D(x), FASC Score, and Baseline comparison metrics."""
        if isinstance(feature_vector, np.ndarray):
            x = torch.from_numpy(feature_vector.astype(np.float32))
        else:
            x = feature_vector.float()

        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        # 1. Full Model Forward Pass
        logits_all = self.model(x)
        probs_all = F.softmax(logits_all, dim=-1).squeeze(0).cpu().numpy()

        # Predicted class y_hat and Base Confidence C(x)
        y_hat_idx = int(np.argmax(probs_all))
        y_hat_name = self.class_names[y_hat_idx]
        c_x = float(probs_all[y_hat_idx])  # Maximum Softmax Probability (MSP)

        # Baseline 2: Top-Two Probability Margin
        sorted_probs = np.sort(probs_all)[::-1]
        top_two_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        # 2. Branch-Ablation Sensitivity: Remove each branch one by one
        branch_sensitivities = {}
        branch_probs = {}
        agreement_count = 0

        total_diff = 0.0
        num_branches = len(BRANCH_SLICES)

        for branch_name, (start_idx, end_idx) in BRANCH_SLICES.items():
            # Create ablated copy with branch zeroed out
            x_ablated = x.clone()
            x_ablated[:, start_idx:end_idx] = 0.0

            logits_m = self.model(x_ablated)
            probs_m = F.softmax(logits_m, dim=-1).squeeze(0).cpu().numpy()

            # Probability assigned to the same class y_hat by the ablated configuration
            p_m_yhat = float(probs_m[y_hat_idx])
            diff = abs(c_x - p_m_yhat)
            total_diff += diff

            branch_sensitivities[branch_name] = round(diff, 4)
            branch_probs[branch_name] = round(p_m_yhat, 4)

            # Check if ablated branch still agrees on the same winning class
            if int(np.argmax(probs_m)) == y_hat_idx:
                agreement_count += 1

        # D(x): Mean sensitivity across the 4 branch ablations
        d_x = total_diff / float(num_branches)

        # Baseline 3: Feature-configuration agreement score
        agreement_score = agreement_count / float(num_branches)

        # Proposed FASC Reliability Score: FASC(x) = C(x) * [1 - D(x)]
        fasc_score = c_x * (1.0 - d_x)
        fasc_score = float(np.clip(fasc_score, 0.0, 1.0))

        # Decision Rule: Accept if FASC >= threshold, otherwise Abstain (Uncertain / Unknown)
        if fasc_score >= self.reliability_threshold:
            decision = "ACCEPT"
            final_verdict = y_hat_name
        else:
            decision = "ABSTAIN"
            final_verdict = "Unknown / Uncertain Generator"

        return {
            "predicted_class": y_hat_name,
            "final_verdict": final_verdict,
            "decision": decision,
            "confidence_score": round(c_x, 4),               # Baseline 1: Standard Confidence C(x)
            "top_two_margin": round(top_two_margin, 4),       # Baseline 2: Margin
            "branch_agreement": round(agreement_score, 4),    # Baseline 3: Agreement
            "ablation_sensitivity_Dx": round(d_x, 4),         # D(x)
            "fasc_reliability_score": round(fasc_score, 4),   # Proposed FASC Score
            "branch_details": {
                "ablated_probabilities": branch_probs,
                "individual_sensitivities": branch_sensitivities,
            },
        }
