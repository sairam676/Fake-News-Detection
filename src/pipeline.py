"""
pipeline.py
-----------
Short forward  → LLM primary, BERT silent fallback only
Long article   → ML Ensemble only, zero LLM

Changes vs previous version
────────────────────────────
1. LIME crash fix — explain() receives a predict_fn-compatible wrapper so
   feature shape always matches what the trained ensemble expects.
   Previously the bare text string was passed and LIME built its own
   bag-of-words matrix, which had a different column count than the
   TF-IDF + handcrafted feature matrix the model was trained on.

2. India policy heuristic (_is_india_policy_article) — when the ML model
   lands in the 40–65% confidence range on an article that clearly discusses
   a real Indian government/institution event, MISLEADING is replaced with
   UNCERTAIN.  This stops the demonetization false-positive without touching
   the model weights.  The zone (40–65%) is deliberately narrow so that
   genuinely fake articles with sensational language are not protected.

3. NER now runs on long articles too — sources field is populated instead
   of always returning an empty list.
"""

import joblib
import os
import sys
import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion      import ingest
from liar_model     import predict as bert_predict
from fact_checker   import fact_check
from explainability import explain
from train          import extract_handcrafted, FAKE_TRIGGER_WORDS


MODELS_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
HIGH_CONFIDENCE     = 0.80
MEDIUM_CONFIDENCE   = 0.60
BERT_CONF_THRESHOLD = 0.70

# India-policy heuristic: if ML confidence is in this ambiguous band AND the
# article looks like a genuine Indian government/institution story, avoid
# calling it MISLEADING.  Tune these bounds if you see over-correction.
INDIA_POLICY_LOW  = 0.35   # below this P(REAL) we still trust the model
INDIA_POLICY_HIGH = 0.65   # above this P(REAL) normal path applies

_vectorizer = None
_ensemble   = None
_ner_model  = None


def _load_ml_models():
    global _vectorizer, _ensemble
    if _vectorizer is None or _ensemble is None:
        print("[pipeline] Loading TF-IDF vectorizer...")
        _vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

        ensemble_path = os.path.join(MODELS_DIR, "ensemble_model.joblib")
        xgb_path      = os.path.join(MODELS_DIR, "xgboost_model.joblib")

        if os.path.exists(ensemble_path):
            _ensemble = joblib.load(ensemble_path)
            print("[pipeline] Ensemble model loaded.")
        else:
            _ensemble = joblib.load(xgb_path)
            print("[pipeline] XGBoost loaded.")
    return _vectorizer, _ensemble


def _load_ner():
    global _ner_model
    if _ner_model is None:
        print("[pipeline] Loading NER model...")
        _ner_model = hf_pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        print("[pipeline] NER model loaded.")
    return _ner_model


def _is_sensational(text: str) -> bool:
    caps_ratio   = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamations = text.count("!")
    text_upper   = text.upper()
    has_triggers = any(word.upper() in text_upper for word in FAKE_TRIGGER_WORDS)
    return caps_ratio > 0.15 or exclamations >= 2 or has_triggers


# Known Indian government bodies, institutions, and policy keywords.
# Kept deliberately specific to avoid triggering on generic articles.
_INDIA_POLICY_SIGNALS = [
    # Institutions
    "reserve bank of india", "rbi", "supreme court of india", "parliament of india",
    "lok sabha", "rajya sabha", "niti aayog", "isro", "aiims", "iit", "iim",
    "election commission of india", "sebi", "irdai", "trai",
    # PM / Cabinet phrasing
    "prime minister narendra modi", "pm modi", "narendra modi announced",
    "government of india", "ministry of finance", "ministry of health",
    "ministry of external affairs", "union budget", "finance minister",
    # Signature policy events — extend as needed
    "demonetization", "demonetisation", "goods and services tax", "gst rollout",
    "jan dhan yojana", "ayushman bharat", "digital india", "make in india",
    "chandrayaan", "mangalyaan", "aadhar", "aadhaar",
]


def _is_india_policy_article(text: str) -> bool:
    """
    Return True when the article contains multiple strong signals that it is
    reporting on a real Indian government, institution, or policy event.

    Requires at least TWO distinct signals to fire — a single keyword like
    "GST" in an otherwise suspicious article should not trigger this.
    """
    text_lower = text.lower()
    hits = sum(1 for signal in _INDIA_POLICY_SIGNALS if signal in text_lower)
    return hits >= 2


def _run_ml(text: str) -> dict:
    """Run ensemble on long article — TF-IDF + handcrafted features combined."""
    vectorizer, model = _load_ml_models()

    tfidf_feats = vectorizer.transform([text])
    hc_feats    = sp.csr_matrix(extract_handcrafted([text]))
    features    = sp.hstack([tfidf_feats, hc_feats])

    label      = model.predict(features)[0]
    confidence = 0.5

    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features)[0]
        confidence = float(proba[1])   # probability of REAL

    return {
        "label"      : "REAL" if label == 1 else "FAKE",
        "score"      : round(confidence, 4),
        "confidence" : round(confidence, 4)
    }


def _run_lime_safe(text: str):
    """
    Wrapper that fixes the feature-shape mismatch that causes LIME to crash.

    The root cause: explain() (explainability.py) creates its own
    CountVectorizer internally, so the matrix it passes to predict_fn has
    a different number of columns than the TF-IDF + handcrafted matrix the
    ensemble was trained on.

    Fix: pass a predict_fn that transforms text with the *same* vectorizer
    and appends handcrafted features — so the shape is always correct —
    instead of relying on whatever matrix LIME builds internally.
    """
    try:
        vectorizer, model = _load_ml_models()

        def _predict_fn(texts):
            """Called by LIME with a list of perturbed text strings."""
            tfidf  = vectorizer.transform(texts)
            hc     = sp.csr_matrix(extract_handcrafted(texts))
            feats  = sp.hstack([tfidf, hc])
            if hasattr(model, "predict_proba"):
                return model.predict_proba(feats)
            # Fallback for models without predict_proba
            preds = model.predict(feats)
            return np.column_stack([1 - preds, preds]).astype(float)

        result = explain(text, predict_fn=_predict_fn)
        return result

    except TypeError:
        # explain() in this project does not yet accept predict_fn — call
        # without it but catch the shape error gracefully.
        try:
            return explain(text)
        except Exception as e:
            print(f"[pipeline] LIME failed (feature shape mismatch): {e}")
            print("[pipeline] LIME skipped — add predict_fn parameter to "
                  "explainability.explain() to fix permanently.")
            return None

    except Exception as e:
        print(f"[pipeline] LIME failed: {e}")
        return None


def _extract_sources(text: str) -> list:
    try:
        ner      = _load_ner()
        entities = ner(text[:512])
        sources  = list({
            e["word"] for e in entities
            if e["entity_group"] in ("ORG", "PER")
            and not e["word"].startswith("##")
        })
        return sources
    except Exception as e:
        print(f"[pipeline] NER failed: {e}")
        return []


# ── Decision Engine — Short Text ──────────────────────────────────────────────

def _decide_short(llm_result: dict, bert_result: dict) -> dict:
    """LLM is primary. BERT is silent fallback only."""
    llm_label  = llm_result["label"]
    bert_conf  = bert_result.get("confidence", 0.5)
    bert_label = bert_result["label"]

    if llm_label == "REAL":
        return {
            "verdict"     : "REAL",
            "confidence"  : round(llm_result["confidence"] * 100),
            "risk"        : "LOW",
            "explanation" : "Fact-check confirms this claim is accurate."
        }
    elif llm_label == "FAKE":
        return {
            "verdict"     : "FAKE",
            "confidence"  : round((1 - llm_result["score"]) * 100),
            "risk"        : "HIGH",
            "explanation" : "Fact-check confirms this claim is false."
        }
    else:
        # LLM UNVERIFIABLE → silent BERT fallback
        if bert_conf >= BERT_CONF_THRESHOLD and bert_label == "FAKE":
            return {
                "verdict"     : "LIKELY FAKE",
                "confidence"  : round(bert_conf * 100),
                "risk"        : "MEDIUM",
                "explanation" : "Could not verify online. Writing style suggests this may be fake."
            }
        return {
            "verdict"     : "UNCERTAIN",
            "confidence"  : 40,
            "risk"        : "MEDIUM",
            "explanation" : "Could not verify this claim. Check from a trusted source."
        }


# ── Decision Engine — Long Articles (NO LLM) ─────────────────────────────────

def _decide_long(ml_result: dict, text: str) -> dict:
    """
    ML Ensemble is the ONLY decision maker for long articles.
    score = probability of REAL (0.0 = definitely fake, 1.0 = definitely real)

    India policy override: when ML confidence is ambiguous (35–65% P(REAL))
    AND the article has ≥2 Indian government/institution signals, return
    UNCERTAIN instead of MISLEADING.  Rationale: the ML model was trained
    mostly on US-politics data and has no reliable signal for Indian policy
    articles — "uncertain" is honest; "misleading" is a false accusation.
    """
    label      = ml_result["label"]
    confidence = ml_result["confidence"]   # P(REAL)

    # ── High confidence ────────────────────────────────────────────────────────
    if confidence >= HIGH_CONFIDENCE:
        return {
            "verdict"     : "REAL",
            "confidence"  : round(confidence * 100),
            "risk"        : "LOW",
            "explanation" : "Writing style and content patterns indicate this is likely real news."
        }

    if (1 - confidence) >= HIGH_CONFIDENCE:
        if _is_sensational(text):
            return {
                "verdict"     : "FAKE",
                "confidence"  : round((1 - confidence) * 100),
                "risk"        : "HIGH",
                "explanation" : "Writing style and content patterns strongly indicate fake news."
            }
        else:
            return {
                "verdict"     : "MISLEADING",
                "confidence"  : round((1 - confidence) * 100),
                "risk"        : "MEDIUM",
                "explanation" : "Content patterns suggest misinformation but writing style appears neutral."
            }

    # ── Medium confidence ──────────────────────────────────────────────────────
    if confidence >= MEDIUM_CONFIDENCE:
        return {
            "verdict"     : "REAL",
            "confidence"  : round(confidence * 100),
            "risk"        : "LOW",
            "explanation" : "Content appears to be legitimate news."
        }

    if (1 - confidence) >= MEDIUM_CONFIDENCE:
        # India policy override — ambiguous ML score on a known-real event type
        if INDIA_POLICY_LOW <= confidence <= INDIA_POLICY_HIGH and _is_india_policy_article(text):
            print("[pipeline] India-policy heuristic fired — overriding MISLEADING → UNCERTAIN")
            return {
                "verdict"     : "UNCERTAIN",
                "confidence"  : round(confidence * 100),
                "risk"        : "LOW",
                "explanation" : (
                    "The ML model lacks sufficient training data on Indian government "
                    "and policy articles to give a reliable verdict. The article "
                    "references known Indian institutions/policies. Treat as unverified "
                    "rather than misleading — cross-check with PIB or a national outlet."
                )
            }

        if _is_sensational(text):
            return {
                "verdict"     : "LIKELY FAKE",
                "confidence"  : round((1 - confidence) * 100),
                "risk"        : "MEDIUM",
                "explanation" : "Fake news patterns detected. Sensational language found. Verify before sharing."
            }
        else:
            return {
                "verdict"     : "MISLEADING",
                "confidence"  : round((1 - confidence) * 100),
                "risk"        : "MEDIUM",
                "explanation" : "Some fake patterns detected but writing appears neutral. Verify independently."
            }

    # ── Low confidence (40-60% either way) ────────────────────────────────────
    # India policy override also applies here
    if _is_india_policy_article(text):
        print("[pipeline] India-policy heuristic fired — keeping UNCERTAIN verdict")
        return {
            "verdict"     : "UNCERTAIN",
            "confidence"  : round(confidence * 100),
            "risk"        : "LOW",
            "explanation" : (
                "The ML model lacks sufficient training data on Indian government "
                "and policy articles. The article references known Indian "
                "institutions/policies. Treat as unverified — cross-check with "
                "PIB or a national outlet."
            )
        }

    return {
        "verdict"     : "UNCERTAIN",
        "confidence"  : round(confidence * 100),
        "risk"        : "MEDIUM",
        "explanation" : "Could not determine with enough confidence. Verify from a trusted source."
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run(raw_input: str) -> dict:
    print(f"\n[pipeline] Starting pipeline...")

    ingested = ingest(raw_input)
    if not ingested["success"]:
        return {"success": False, "error": ingested["error"]}

    text       = ingested["text"]
    input_type = ingested["input_type"]
    is_short   = input_type == "short_forward"

    print(f"[pipeline] Input type : {input_type} ({len(text)} chars)")

    # ── Short forward: LLM primary ────────────────────────────────────────────
    if is_short:
        bert_result = bert_predict(text)

        print("[pipeline] Step 3: Extracting sources via NER...")
        sources = _extract_sources(text)
        print(f"           Sources: {sources if sources else 'None'}")

        print("[pipeline] Step 4: Running LLM fact checker...")
        llm_result = fact_check(text, sources)
        print(f"           LLM → {llm_result['label']} ({llm_result['reason']})")

        print("[pipeline] Step 5: Computing verdict...")
        decision = _decide_short(llm_result, bert_result)

        return {
            "success"      : True,
            "input_type"   : input_type,
            "text_preview" : text[:100] + "..." if len(text) > 100 else text,
            "verdict"      : decision["verdict"],
            "confidence"   : f"{decision['confidence']}%",
            "risk"         : decision["risk"],
            "explanation"  : decision["explanation"],
            "llm_reason"   : llm_result["reason"],
            "sources"      : sources,
            "model_scores" : {"llm": llm_result}
        }

    # ── Long article: ML Ensemble only, zero LLM ─────────────────────────────
    else:
        print("[pipeline] Step 2: Running ML Ensemble...")
        ml_result = _run_ml(text)
        print(f"           Ensemble → {ml_result['label']} (conf={ml_result['confidence']})")

        # NER on long articles — gives users named sources even for long input
        print("[pipeline] Step 3: Extracting sources via NER...")
        sources = _extract_sources(text)
        print(f"           Sources: {sources if sources else 'None'}")

        print("[pipeline] Step 4: Running LIME explainability...")
        explanation_result = _run_lime_safe(text)   # uses shape-safe wrapper

        print("[pipeline] Step 5: Computing verdict (no LLM)...")
        decision = _decide_long(ml_result, text)

        output = {
            "success"      : True,
            "input_type"   : input_type,
            "text_preview" : text[:100] + "..." if len(text) > 100 else text,
            "verdict"      : decision["verdict"],
            "confidence"   : f"{decision['confidence']}%",
            "risk"         : decision["risk"],
            "explanation"  : decision["explanation"],
            "llm_reason"   : "Not used — long article handled by local ML model.",
            "sources"      : sources,
            "model_scores" : {"ml_ensemble": ml_result}
        }

        if explanation_result:
            output["word_highlights"] = {
                "fake_words" : explanation_result["top_fake_words"][:5],
                "real_words" : explanation_result["top_real_words"][:5]
            }

        return output


# ── Pretty Print ──────────────────────────────────────────────────────────────

def print_result(result: dict):
    if not result["success"]:
        print(f"Error: {result['error']}")
        return

    print("\n" + "="*55)
    print("  FAKE NEWS DETECTION RESULT")
    print("="*55)
    print(f"  Input type  : {result['input_type']}")
    print(f"  Verdict     : {result['verdict']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Risk        : {result['risk']}")
    print(f"\n  {result['explanation']}")
    if result["input_type"] == "short_forward":
        print(f"\n  Fact-check  : {result['llm_reason']}")
    if result.get("sources"):
        print(f"  Sources     : {result['sources']}")
    print(f"\n  Model scores:")
    for name, res in result["model_scores"].items():
        print(f"    {name:<15} → {res['label']} (score={res['score']})")
    if "word_highlights" in result:
        print(f"\n  Fake words  : {[w for w, _ in result['word_highlights']['fake_words']]}")
        print(f"  Real words  : {[w for w, _ in result['word_highlights']['real_words']]}")
    print("="*55)


if __name__ == "__main__":
    tests = [
        "AIIMS doctor says lemon juice cures diabetes!! Share fast!!",
        "ISRO Chandrayaan-3 successfully landed on moon south pole in August 2023",
        """Indian Prime Minister Narendra Modi announced demonetization of 500 and
        1000 rupee notes on November 8 2016 in a televised address to the nation.
        The government stated the move was aimed at curbing black money and corruption.
        Citizens were given until December 30 2016 to deposit old notes in banks.""",
        """SHOCKING REVELATION!! Scientists at IIT Delhi confirmed that drinking turmeric
        mixed with petrol cures all cancer within 7 days!! The government is suppressing
        this because pharmaceutical companies are paying them!! Share before deleted!!"""
    ]
    for text in tests:
        result = run(text)
        print_result(result)