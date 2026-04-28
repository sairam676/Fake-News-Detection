"""
liar_model.py
-------------
Short claim classifier using a pretrained HuggingFace model.
No training needed — downloads once, runs locally forever.

Best for: short WhatsApp forwards under 280 chars.

Model: hamzab/roberta-fake-news-classification
"""

import os
from transformers import pipeline


MODEL_NAME  = "hamzab/roberta-fake-news-classification"
SAVE_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "bert_liar_model")
MAX_LENGTH  = 128

_classifier = None


def _load_model():
    global _classifier
    if _classifier is None:
        if os.path.exists(SAVE_PATH):
            print("[liar_model] Loading from models/bert_liar_model/...")
            _classifier = pipeline(
                "text-classification",
                model=SAVE_PATH,
                truncation=True,
                max_length=MAX_LENGTH
            )
        else:
            print("[liar_model] Downloading from HuggingFace (first time only)...")
            _classifier = pipeline(
                "text-classification",
                model=MODEL_NAME,
                truncation=True,
                max_length=MAX_LENGTH
            )
            os.makedirs(SAVE_PATH, exist_ok=True)
            _classifier.model.save_pretrained(SAVE_PATH)
            _classifier.tokenizer.save_pretrained(SAVE_PATH)
            print(f"[liar_model] Model saved to models/bert_liar_model/")
        print("[liar_model] Model ready.")


def predict(text: str) -> dict:
    """
    Classify a short claim as FAKE or REAL.

    Args:
        text: Short claim text (WhatsApp forward)

    Returns:
        {
            "label"      : "FAKE" or "REAL",
            "score"      : float 0-1 (probability of REAL),
            "confidence" : float 0-1
        }
    """
    _load_model()

    result     = _classifier(text[:512])[0]
    label      = result["label"].upper()
    confidence = round(result["score"], 4)

    if "REAL" in label or label == "LABEL_1":
        normalized = "REAL"
        score      = confidence
    else:
        normalized = "FAKE"
        score      = round(1.0 - confidence, 4)

    return {
        "label"      : normalized,
        "score"      : score,
        "confidence" : confidence
    }


if __name__ == "__main__":
    test_claims = [
        "AIIMS doctor says lemon juice cures diabetes!! Share fast!!",
        "The government launched a new scheme for farmers this quarter.",
        "Scientists confirm 5G towers spread coronavirus.",
        "India GDP grew by 7.2 percent last quarter according to official data.",
    ]

    for claim in test_claims:
        result = predict(claim)
        print(f"Claim      : {claim[:65]}")
        print(f"Label      : {result['label']}")
        print(f"Score      : {result['score']}")
        print(f"Confidence : {result['confidence']}")
        print()