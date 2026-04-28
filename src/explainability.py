"""
explainability.py
-----------------
Uses LIME to explain WHY the ML model predicted FAKE or REAL.
Shows which words pushed the prediction in each direction.

Only works with the local TF-IDF + ML model (not BERT or LLM).
Used for long articles where TF-IDF runs.

Output:
    List of (word, weight) pairs
    Positive weight → pushed toward REAL
    Negative weight → pushed toward FAKE
"""

import numpy as np
import joblib
import os
from lime.lime_text import LimeTextExplainer


# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
NUM_FEATURES = 10    # top 10 words to highlight
NUM_SAMPLES  = 500   # LIME samples — lower = faster, higher = more accurate

CLASS_NAMES  = ["FAKE", "REAL"]


# ── Load Models ───────────────────────────────────────────────────────────────

def _load_models():
    """Load vectorizer and ML model for explanation."""
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    # Use XGBoost for LIME — it supports predict_proba and works with current feature count
    # Fall back to ensemble if XGBoost not found
    xgb_path      = os.path.join(MODELS_DIR, "xgboost_model.joblib")
    ensemble_path = os.path.join(MODELS_DIR, "ensemble_model.joblib")

    vectorizer = joblib.load(vectorizer_path)

    if os.path.exists(xgb_path):
        model = joblib.load(xgb_path)
    else:
        model = joblib.load(ensemble_path)

    return vectorizer, model


# ── Prediction Function For LIME ──────────────────────────────────────────────

def _make_predict_fn(vectorizer, model):
    """
    Returns a prediction function LIME can call.
    Combines TF-IDF + handcrafted features to match training feature shape.
    """
    import scipy.sparse as sp
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import extract_handcrafted

    def predict_proba(texts):
        tfidf    = vectorizer.transform(texts)
        hc       = sp.csr_matrix(extract_handcrafted(texts))
        features = sp.hstack([tfidf, hc])
        return model.predict_proba(features)

    return predict_proba


# ── Explain ───────────────────────────────────────────────────────────────────

def explain(text: str) -> dict:
    """
    Explain why the ML model made its prediction.

    Args:
        text: Article or forward text

    Returns:
        {
            "top_fake_words" : [(word, weight), ...],  top words pushing FAKE
            "top_real_words" : [(word, weight), ...],  top words pushing REAL
            "all_words"      : [(word, weight), ...],  all features with weights
            "prediction"     : "FAKE" or "REAL",
            "confidence"     : float
        }
    """
    vectorizer, model = _load_models()
    predict_fn        = _make_predict_fn(vectorizer, model)

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features = NUM_FEATURES,
        num_samples  = NUM_SAMPLES,
        labels       = (0, 1)   # 0=FAKE, 1=REAL
    )

    # Get prediction on original text
    features   = vectorizer.transform([text])
    proba      = model.predict_proba(features)[0]
    pred_label = CLASS_NAMES[np.argmax(proba)]
    confidence = round(float(np.max(proba)), 4)

    # Extract word weights
    # Positive weight for label 1 (REAL) = pushes toward REAL
    # Negative weight for label 1 (REAL) = pushes toward FAKE
    word_weights = explanation.as_list(label=1)

    fake_words = [
        (word, round(abs(weight), 4))
        for word, weight in word_weights
        if weight < 0
    ]

    real_words = [
        (word, round(weight, 4))
        for word, weight in word_weights
        if weight > 0
    ]

    # Sort by weight descending
    fake_words.sort(key=lambda x: x[1], reverse=True)
    real_words.sort(key=lambda x: x[1], reverse=True)

    return {
        "prediction"     : pred_label,
        "confidence"     : confidence,
        "top_fake_words" : fake_words,
        "top_real_words" : real_words,
        "all_words"      : word_weights
    }


def format_explanation(result: dict) -> str:
    """
    Format explanation as readable text for WhatsApp reply or web UI.
    """
    lines = []
    lines.append(f"Prediction : {result['prediction']} ({result['confidence']*100:.1f}% confident)")
    lines.append("")

    if result["top_fake_words"]:
        lines.append("Words that suggest FAKE:")
        for word, weight in result["top_fake_words"][:5]:
            bar = "█" * int(weight * 50)
            lines.append(f"  {word:<20} {bar}")

    if result["top_real_words"]:
        lines.append("")
        lines.append("Words that suggest REAL:")
        for word, weight in result["top_real_words"][:5]:
            bar = "█" * int(weight * 50)
            lines.append(f"  {word:<20} {bar}")

    return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_texts = [
        "SHOCKING!! Government HIDING vaccine side effects from citizens!! Share before deleted!!",
        "The Reserve Bank of India announced a 25 basis point rate cut in its quarterly policy meeting held on Friday."
    ]

    for text in test_texts:
        print(f"\nText: {text[:70]}...")
        print("-" * 50)
        result = explain(text)
        print(format_explanation(result))
        print()