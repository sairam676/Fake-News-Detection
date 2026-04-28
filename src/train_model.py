"""
train_model.py
--------------
Trains XGBoost + Random Forest ensemble.
XGBoost alone = 97.53%, XGB+RF ensemble = 97.33%

Models saved:
    models/tfidf_vectorizer.joblib
    models/xgboost_model.joblib
    models/random_forest.joblib
    models/ensemble_model.joblib  ← pipeline uses this
"""

import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics  import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("[train_model] XGBoost not installed. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
RANDOM_STATE = 42


def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer):
    """
    Train XGBoost + RF + Ensemble on pre-built feature matrices.
    Accepts features from train_pipeline.py — does NOT reload data internally.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    results = {}
    xgb = None

    # ── XGBoost ────────────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        print("\n" + "="*50 + "\nTraining XGBoost\n" + "="*50)
        xgb = XGBClassifier(
            n_estimators     = 300,
            max_depth        = 7,
            learning_rate    = 0.08,
            subsample        = 0.85,
            colsample_bytree = 0.85,
            min_child_weight = 3,
            eval_metric      = "logloss",
            random_state     = RANDOM_STATE,
            n_jobs           = -1
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
        results["XGBoost"] = acc

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("\n" + "="*50 + "\nTraining Random Forest\n" + "="*50)
    rf = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 25,
        min_samples_leaf = 2,
        n_jobs           = -1,
        random_state     = RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    results["Random Forest"] = acc

    # ── Ensemble: XGBoost + RF ─────────────────────────────────────────────────
    print("\n" + "="*50 + "\nTraining Voting Ensemble\n" + "="*50)
    estimators = [("xgb", xgb), ("rf", rf)] if xgb else [("rf", rf)]
    ensemble   = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    acc_ens    = accuracy_score(y_test, y_pred_ens)
    print(f"\nAccuracy: {acc_ens:.4f} ({acc_ens*100:.2f}%)")
    print(classification_report(y_test, y_pred_ens, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ens))
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "ensemble_model.joblib"))
    results["Ensemble (XGB+RF)"] = acc_ens

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        tag = "  ← pipeline uses this" if "Ensemble" in name else ""
        print(f"  {name:<25} → {acc*100:.2f}%{tag}")
    print(f"\nAll models saved to {MODELS_DIR}")
    print("Best model for pipeline: Ensemble")