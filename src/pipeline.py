"""
pipeline.py  — FINAL VERSION
-----------------------------
All fixes applied:
  1. _detect_category: correct order fake_pattern → health → politics → out_of_scope
     + expanded FAKE_PATTERNS to match real WhatsApp forwards
  2. _short_text_rule_check: fires for both "health" and "fake_pattern" categories
  3. LIME noise words: stopwords + Indian proper nouns filtered from word_highlights
  4. word_highlights: len > 2 filter kills single/two-char tokens
  5. US news separate verdict path — not penalised by Indian corpus bias
  6. REAL verdict now possible at moderate confidence (was missing before)
  7. liar_model blend only for short text < 300 chars
"""

import joblib, os, sys
import scipy.sparse as sp
from transformers import pipeline as hf_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion      import ingest
from explainability import explain
from train          import extract_handcrafted, FAKE_TRIGGER_WORDS

MODELS_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
HIGH_CONFIDENCE   = 0.75
MEDIUM_CONFIDENCE = 0.55

_vectorizer = None
_ensemble   = None
_ner_model  = None


# ── Scope keywords ─────────────────────────────────────────────────────────────

HEALTH_KEYWORDS = [
    "doctor", "hospital", "vaccine", "virus", "covid", "disease", "cure",
    "cures", "medicine", "health", "aiims", "who", "treatment", "drug",
    "cancer", "diabetes", "symptom", "patient", "medical", "tablet", "dose",
    "clinical", "remedy", "ayurveda", "homeopathy", "immunity", "infection",
    "bacteria", "oxygen", "icu", "surgery", "nurse", "pharmacy", "chemist",
    "injection", "blood", "heart", "kidney", "liver", "lungs", "fever",
    "cough", "cold", "lemon water", "lemon juice", "turmeric", "neem",
    "drinking", "eat this", "home remedy", "ancient remedy", "miracle",
]

POLITICS_KEYWORDS = [
    "government", "minister", "parliament", "election", "modi", "bjp",
    "congress", "policy", "president", "prime minister", "vote", "political",
    "opposition", "trump", "biden", "senator", "democrat", "republican",
    "narendra", "rahul", "kejriwal", "court", "judiciary", "rbi", "isro",
    "rupee", "demonetization", "scheme", "budget", "cabinet", "lok sabha",
    "rajya sabha", "chief minister", "governor", "law", "act", "bill",
    "india", "indian", "pakistan", "china", "nation", "national",
    "announced", "launched", "declared", "inaugurated", "appointed",
    "supreme court", "high court", "cbi", "ed", "income tax", "gst",
    "aadhaar", "upi", "digital india", "yojana", "pradhan mantri",
    "cm", "mp", "mla", "party", "coalition", "senate", "impeach",
    "house of representatives", "acquit", "charges", "abuse of power",
    "white house", "pentagon", "legislation", "executive order",
    "fbi", "cia", "nato", "united nations", "sanctions",
]

# Checked FIRST — definitive misinformation signals
FAKE_PATTERNS = [
    "shocking", "share now", "share fast", "forward",
    "viral", "suppressed", "hidden", "they dont want", "wake up",
    "exposed", "deleted", "breaking", "before its too late",
    "mainstream media", "doctors hate", "doctors don't want",
    "cures completely", "cures diabetes", "cures cancer", "cures covid",
    "miracle cure", "secret cure", "government hiding", "government is hiding",
    "share before", "forward to all", "forward to 10", "share before deleted",
    "they don't want you", "100% cure", "one weird trick",
    "ancient secret", "big pharma", "pharma hiding",
    "scientists confirm", "scientists hate", "share with everyone",
    "!!!", "shocking truth", "wake up people", "hate this",
]

# US news markers — separate verdict path, not penalised by Indian corpus bias
US_NEWS_MARKERS = [
    "trump", "biden", "obama", "clinton", "senate",
    "house of representatives", "republican", "democrat", "white house",
    "pentagon", "fbi", "cia", "washington", "new york times", "cnn",
    "fox news", "msnbc", "impeach", "acquit", "ukraine", "nato",
    "midterm", "primary election", "electoral college",
    "justice department", "attorney general", "mueller", "january 6",
]

# LIME noise words — training artifacts + stopwords
# Stops "the", "in", "and", "Modi", "2016" etc. showing as suspicious words
LIME_NOISE_WORDS = {
    # Indian proper nouns / dates that are training artifacts
    'indian', 'india', 'modi', 'narendra', 'november', 'january',
    'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'december', '2016', '2017', '2018', '2019',
    '2020', '2021', '2022', '2023', 'said', 'also', 'new', 'government',
    'minister', 'rupee', 'lakh', 'crore', 'announced', 'launched', 'declared',
    # Stopwords
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'and', 'or',
    'is', 'was', 'are', 'were', 'be', 'been', 'that', 'this', 'it',
    'its', 'for', 'from', 'by', 'with', 'as', 'into', 'has', 'have',
    'had', 'he', 'she', 'they', 'we', 'his', 'her', 'their',
    'not', 'but', 'if', 'about', 'up', 'out', 'so', 'do', 'did',
    'will', 'would', 'could', 'should', 'been', 'being', 'than',
}


# ── Category detection ────────────────────────────────────────────────────────
# Order matters: fake_pattern → health → politics → out_of_scope

def _detect_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in FAKE_PATTERNS):     return "fake_pattern"
    if any(k in t for k in HEALTH_KEYWORDS):   return "health"
    if any(k in t for k in POLITICS_KEYWORDS): return "politics"
    return "out_of_scope"


# ── Model loading ──────────────────────────────────────────────────────────────

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
            print("[pipeline] XGBoost loaded (no ensemble found).")
    return _vectorizer, _ensemble


def _load_ner():
    global _ner_model
    if _ner_model is None:
        _ner_model = hf_pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
    return _ner_model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_sensational(text: str) -> bool:
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return (
        caps_ratio > 0.15
        or text.count("!") >= 2
        or any(w.upper() in text.upper() for w in FAKE_TRIGGER_WORDS)
    )


def _run_ml(text: str) -> dict:
    vectorizer, model = _load_ml_models()
    tfidf = vectorizer.transform([text])
    hc    = sp.csr_matrix(extract_handcrafted([text]))
    feats = sp.hstack([tfidf, hc])
    conf  = float(model.predict_proba(feats)[0][1]) if hasattr(model, "predict_proba") else 0.5

    # Blend with liar_model for short claims only
    if len(text) < 300:
        try:
            from liar_model import predict as liar_predict
            liar = liar_predict(text)
            conf = round((conf + liar["score"]) / 2, 4)
        except Exception as e:
            print(f"[pipeline] liar_model skipped: {e}")

    return {
        "label"      : "REAL" if conf >= 0.5 else "FAKE",
        "score"      : round(conf, 4),
        "confidence" : round(conf, 4),
    }


def _extract_sources(text: str) -> list:
    try:
        ner      = _load_ner()
        entities = ner(text[:512])
        return list({
            e["word"] for e in entities
            if e["entity_group"] in ("ORG", "PER")
            and not e["word"].startswith("##")
        })
    except Exception as e:
        print(f"[pipeline] NER failed: {e}")
        return []


def _run_lime(text: str):
    try:
        return explain(text)
    except Exception as e:
        print(f"[pipeline] LIME failed: {e}")
        return None


def _filter_lime_words(word_score_list: list) -> list:
    """Remove stopwords, noise, and single/two-char tokens from LIME output."""
    return [
        (str(w), float(s))
        for w, s in word_score_list
        if str(w).lower() not in LIME_NOISE_WORDS
        and len(str(w)) > 2
    ]


# ── Rule-based override for short text ────────────────────────────────────────

SHORT_TEXT_THRESHOLD = 300

STRONG_FAKE_HEALTH_CLAIMS = [
    "cures diabetes", "cures cancer", "cures covid", "cures all",
    "cure for cancer", "cure for diabetes", "destroys cancer",
    "doctors hate", "they dont want you to know",
    "100% cure", "miracle cure", "home remedy cures", "lemon juice cures",
    "lemon water cures", "turmeric cures", "one trick", "ancient remedy",
    "secret cure", "government hiding", "government is hiding",
    "government suppressing", "mainstream media hiding",
    "they are hiding the cure", "scientists confirm 5g",
    "5g towers spread", "5g causes", "vaccine causes infertility",
    "vaccine causes cancer", "chips in vaccine", "microchip vaccine",
    "share before deleted", "forward to 10", "forward to all",
    "doctors dont want", "pharma companies hiding",
    "drinking lemon water", "cures completely",
]

SUPPRESSION_SIGNALS = [
    "suppressing", "suppressed", "hiding", "hidden", "before deleted",
    "share fast", "share now", "forward this", "mainstream media",
    "they dont want", "before its too late", "going viral",
]


def _short_text_rule_check(text: str, category: str) -> dict | None:
    if len(text) > SHORT_TEXT_THRESHOLD:
        return None

    t_low = text.lower()

    # Rule 1: Medically impossible claims — fires for health AND fake_pattern
    if category in ("health", "fake_pattern"):
        for claim in STRONG_FAKE_HEALTH_CLAIMS:
            if claim in t_low:
                return {
                    "verdict"    : "FAKE",
                    "confidence" : 95,
                    "risk"       : "HIGH",
                    "explanation": "Contains a medically impossible claim. No scientific evidence supports this.",
                }

    # Rule 2: Sensationalism score
    hc = extract_handcrafted([text])[0]
    caps_ratio        = hc[0]
    exclamation_count = hc[1]
    clickbait_count   = hc[3]
    all_caps_words    = hc[5]
    repeated_punct    = hc[10]

    signal_score = 0
    if caps_ratio > 0.20:       signal_score += 1
    if exclamation_count >= 2:  signal_score += 1
    if clickbait_count >= 2:    signal_score += 2
    if clickbait_count >= 1:    signal_score += 1
    if all_caps_words >= 2:     signal_score += 1
    if repeated_punct >= 1:     signal_score += 1

    if signal_score >= 4:
        return {
            "verdict"    : "FAKE",
            "confidence" : min(90, 70 + signal_score * 3),
            "risk"       : "HIGH",
            "explanation": "Multiple fake news signals detected: sensational language, clickbait words, excessive punctuation.",
        }

    # Rule 3: Moderate sensationalism + health/fake_pattern
    if signal_score >= 1 and category in ("health", "fake_pattern"):
        return {
            "verdict"    : "LIKELY FAKE",
            "confidence" : 72,
            "risk"       : "MEDIUM",
            "explanation": "Health claim with suspicious language. No credible medical source. Verify before sharing.",
        }

    # Rule 4: Politics + suppression language
    if category == "politics" and any(s in t_low for s in SUPPRESSION_SIGNALS):
        return {
            "verdict"    : "LIKELY FAKE",
            "confidence" : 78,
            "risk"       : "MEDIUM",
            "explanation": "Political claim with suppression/urgency language — common fake news pattern. Verify with PIB or a national outlet.",
        }

    return None


# ── Decision engine ────────────────────────────────────────────────────────────

def _decide(ml_result: dict, text: str, category: str) -> dict:
    conf  = ml_result["confidence"]
    pfake = 1 - conf
    sensational = _is_sensational(text)
    t_low = text.lower()

    INDIAN_MARKERS = [
        "rupee", "lakh", "crore", "modi", "bjp", "lok sabha", "rajya sabha",
        "demonetization", "demonetisation", "rbi", "isro", "aadhaar", "upi",
        "reserve bank", "supreme court", "electoral", "repo rate", "niti",
    ]
    is_indian  = any(m in t_low for m in INDIAN_MARKERS)
    is_us_news = any(m in t_low for m in US_NEWS_MARKERS)

    # Out of scope — return immediately
    if category == "out_of_scope":
        return {
            "verdict"    : "OUT OF SCOPE",
            "confidence" : 0,
            "risk"       : "UNKNOWN",
            "explanation": (
                "This content is not in a category the model was trained on. "
                "Supported: Indian/US politics, health claims, misinformation patterns. "
                "Cannot classify: sports, entertainment, finance."
            ),
        }

    # ── Strong REAL (conf >= 0.75) ─────────────────────────────────────────────
    if conf >= HIGH_CONFIDENCE:
        return {
            "verdict"    : "REAL",
            "confidence" : round(conf * 100),
            "risk"       : "LOW",
            "explanation": "Content patterns match legitimate news. Likely real.",
        }

    # ── Strong FAKE (pfake >= 0.75) ────────────────────────────────────────────
    if pfake >= HIGH_CONFIDENCE:
        if sensational:
            return {
                "verdict"    : "FAKE",
                "confidence" : round(pfake * 100),
                "risk"       : "HIGH",
                "explanation": "Strong fake news signals detected. Sensational language confirmed.",
            }
        # Indian political real news looks fake to the model — don't over-call it
        if is_indian or (category == "politics" and not is_us_news):
            return {
                "verdict"    : "UNCERTAIN",
                "confidence" : round(pfake * 100),
                "risk"       : "MEDIUM",
                "explanation": "Model is uncertain. Appears to be a factual political statement — verify with PIB Fact Check or a national outlet.",
            }
        if is_us_news:
            return {
                "verdict"    : "MISLEADING",
                "confidence" : round(pfake * 100),
                "risk"       : "HIGH",
                "explanation": "Content patterns strongly suggest misinformation. Verify with Reuters or AP News.",
            }
        return {
            "verdict"    : "MISLEADING",
            "confidence" : round(pfake * 100),
            "risk"       : "HIGH",
            "explanation": "Content patterns strongly suggest misinformation. Writing appears deliberately neutral.",
        }

    # ── Moderate REAL (conf >= 0.55) ───────────────────────────────────────────
    if conf >= MEDIUM_CONFIDENCE:
        return {
            "verdict"    : "REAL",
            "confidence" : round(conf * 100),
            "risk"       : "LOW",
            "explanation": "Content leans real. Verify from a trusted source before sharing.",
        }

    # ── Moderate FAKE (pfake >= 0.55) ──────────────────────────────────────────
    if pfake >= MEDIUM_CONFIDENCE:
        if sensational:
            return {
                "verdict"    : "LIKELY FAKE",
                "confidence" : round(pfake * 100),
                "risk"       : "MEDIUM",
                "explanation": "Fake patterns detected with sensational language. Do not share without verifying.",
            }
        return {
            "verdict"    : "UNCERTAIN",
            "confidence" : round(pfake * 100),
            "risk"       : "MEDIUM",
            "explanation": "Model confidence is low. Verify independently with PIB Fact Check or Reuters.",
        }

    # ── Too close to call ──────────────────────────────────────────────────────
    return {
        "verdict"    : "UNCERTAIN",
        "confidence" : round(max(conf, pfake) * 100),
        "risk"       : "MEDIUM",
        "explanation": "Model confidence too low for a reliable verdict. Verify from a trusted source.",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(raw_input: str) -> dict:
    print(f"\n[pipeline] Starting pipeline...")

    ingested = ingest(raw_input)
    if not ingested["success"]:
        return {"success": False, "error": ingested["error"]}

    text       = ingested["text"]
    input_type = ingested["input_type"]
    print(f"[pipeline] Input type : {input_type} ({len(text)} chars)")

    print("[pipeline] Step 1: Detecting category...")
    category = _detect_category(text)
    print(f"           Category: {category}")

    print("[pipeline] Step 2: Running ML Ensemble...")
    ml_result = _run_ml(text)
    print(f"           Ensemble → {ml_result['label']} (conf={ml_result['confidence']})")

    print("[pipeline] Step 3: Extracting sources via NER...")
    sources = _extract_sources(text)
    print(f"           Sources: {sources or 'None'}")

    lime_result = None
    if input_type != "short_forward":
        print("[pipeline] Step 4: Running LIME explainability...")
        lime_result = _run_lime(text)
        if lime_result:
            filtered = _filter_lime_words(lime_result.get("top_fake_words", []))
            print(f"           LIME OK — top fake words: {[w for w, _ in filtered[:3]]}")

    print("[pipeline] Step 5: Computing verdict...")

    rule_decision = None
    if input_type == "short_forward":
        rule_decision = _short_text_rule_check(text, category)
        if rule_decision:
            print(f"           Rules fired → {rule_decision['verdict']} (overrides ML)")

    decision = rule_decision if rule_decision else _decide(ml_result, text, category)

    output = {
        "success"     : True,
        "input_type"  : input_type,
        "text_preview": text[:100] + "..." if len(text) > 100 else text,
        "verdict"     : decision["verdict"],
        "confidence"  : f"{decision['confidence']}%",
        "risk"        : decision["risk"],
        "explanation" : decision["explanation"],
        "sources"     : sources,
        "model_scores": {"ml_ensemble": ml_result},
    }

    if lime_result:
        output["word_highlights"] = {
            "fake_words": _filter_lime_words(lime_result.get("top_fake_words", []))[:5],
            "real_words": _filter_lime_words(lime_result.get("top_real_words", []))[:5],
        }

    return output


# ── Print helper ───────────────────────────────────────────────────────────────

def print_result(result: dict):
    if not result["success"]:
        print(f"Error: {result['error']}")
        return
    print("\n" + "=" * 55)
    print("  FAKE NEWS DETECTION RESULT")
    print("=" * 55)
    print(f"  Input type  : {result['input_type']}")
    print(f"  Verdict     : {result['verdict']}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Risk        : {result['risk']}")
    print(f"\n  {result['explanation']}")
    print(f"\n  Sources     : {result['sources'] or 'None found'}")
    print(f"\n  Model scores:")
    for name, res in result["model_scores"].items():
        print(f"    {name:<15} → {res['label']} (score={res['score']})")
    if "word_highlights" in result:
        fake_words = [w for w, _ in result["word_highlights"]["fake_words"]]
        real_words = [w for w, _ in result["word_highlights"]["real_words"]]
        if fake_words:
            print(f"\n  Fake words  : {fake_words}")
        if real_words:
            print(f"  Real words  : {real_words}")
    print("=" * 55)


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Should be FAKE (rule-based)
        "SHOCKING!!! Doctors HATE this! Drinking lemon water every morning CURES diabetes completely!!!",
        "AIIMS doctor says lemon juice cures diabetes!! Share fast!!",
        "Scientists confirm 5G towers spread coronavirus and government is hiding it!!",
        # Should be LIKELY FAKE
        "Government is suppressing the cure for diabetes. Share this before it gets deleted.",
        # Should be UNCERTAIN (honest — model doesn't know)
        "Donald Trump was impeached by the House of Representatives in December 2019 on charges of abuse of power.",
        # Should be REAL
        """The Indian Space Research Organisation successfully launched its Chandrayaan-3 mission
        on July 14 2023 from the Satish Dhawan Space Centre in Sriharikota. The lander Vikram
        touched down near the lunar south pole on August 23 2023 making India the first country
        to land near the lunar south pole and the fourth country overall to achieve a soft landing.""",
        # Should be OUT OF SCOPE
        "Virat Kohli scores century in test match against Australia at Lords.",
    ]
    for text in tests:
        result = run(text)
        print_result(result)