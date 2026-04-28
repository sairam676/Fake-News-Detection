"""
train.py
--------
Feature preparation for the fake news detector.

Changes from v1:
    - Logistic Regression REMOVED from ensemble (was 58% — was dragging ensemble down)
    - COVID dataset is class-balanced before merging (was 9727:474 — poison)
    - TF-IDF upgraded: sublinear_tf=True, max_features=15000, min_df=2
    - Handcrafted features expanded to 13 signals
    - Domain scope filter: only keep politics / health / India news topics

Best model order (from your results):
    XGBoost alone   → 95.88%   ← use this as primary
    Ensemble XGB+RF → ~97%     ← after balancing fix (expected)
"""

import re
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Domain Scope ───────────────────────────────────────────────────────────────
# Only classify news in these domains — anything else → "Out of scope"
# This prevents the model from giving wrong verdicts on sports/entertainment/etc.

ALLOWED_TOPICS = [
    # Politics
    "government", "minister", "parliament", "election", "modi", "bjp", "congress",
    "policy", "president", "prime minister", "vote", "political", "opposition",
    "trump", "biden", "senator", "democrat", "republican",
    # Health / Medical
    "doctor", "hospital", "vaccine", "virus", "covid", "disease", "cure",
    "medicine", "health", "aiims", "who", "treatment", "drug", "cancer",
    "diabetes", "symptom", "patient", "medical",
    # India specific
    "india", "indian", "isro", "rbi", "rupee", "delhi", "mumbai", "bihar",
    "narendra", "rahul", "kejriwal", "supreme court", "judiciary", "niti aayog",
    # General news / crime / economy
    "police", "crime", "arrest", "court", "judge", "economy", "gdp", "inflation",
    "bank", "finance", "stock", "market", "army", "military", "attack", "protest",
    # Fake news patterns (always in scope)
    "shocking", "share", "forward", "viral", "breaking", "exclusive", "revealed",
    "hidden", "suppressed", "before deleted"
]

ALLOWED_TOPICS_SET = set(t.lower() for t in ALLOWED_TOPICS)


def is_in_scope(text: str) -> bool:
    """Return True if text is about a supported topic."""
    text_lower = text.lower()
    return any(topic in text_lower for topic in ALLOWED_TOPICS_SET)


# ── Clickbait / Fake Trigger Words ────────────────────────────────────────────

FAKE_TRIGGER_WORDS = [
    "shocking", "revealed", "breaking", "share now", "share fast",
    "deleted", "hiding", "hidden", "truth", "wake up", "bombshell",
    "exposed", "cover up", "before its too late", "they dont want",
    "mainstream media", "deep state", "suppressed", "must see",
    "you wont believe", "doctors hate", "miracle", "cure", "secret",
    "urgent", "alert", "exclusive", "leaked", "share before",
    "forward this", "going viral", "100% proven", "scientists shocked"
]


# ── Handcrafted Features ───────────────────────────────────────────────────────

FEATURE_NAMES = [
    "caps_ratio",
    "exclamation_count",
    "question_count",
    "clickbait_word_count",
    "has_clickbait",
    "all_caps_word_count",
    "all_caps_word_ratio",
    "avg_word_length",
    "vocabulary_richness",
    "ellipsis_count",
    "repeated_punctuation",
    "word_count",
    "paragraph_count",
]


def extract_handcrafted(texts) -> np.ndarray:
    """
    Extract 13 handcrafted features per text.
    Works on a pandas Series or list of strings.

    Returns:
        numpy array shape (n_samples, 13)
    """
    rows = []
    for text in texts:
        t     = str(text)
        words = t.split()
        t_low = t.lower()

        n_chars = max(len(t), 1)
        n_words = max(len(words), 1)

        clickbait_count = sum(1 for w in FAKE_TRIGGER_WORDS if w in t_low)

        row = [
            sum(1 for c in t if c.isupper()) / n_chars,           # caps_ratio
            min(t.count("!"), 10),                                  # exclamation_count
            min(t.count("?"), 10),                                  # question_count
            clickbait_count,                                        # clickbait_word_count
            int(clickbait_count > 0),                               # has_clickbait
            sum(1 for w in words if w.isupper() and len(w) > 2),   # all_caps_word_count
            sum(1 for w in words if w.isupper() and len(w) > 2) / n_words,  # all_caps_word_ratio
            sum(len(w) for w in words) / n_words,                  # avg_word_length
            len(set(words)) / n_words,                             # vocabulary_richness
            t.count("..."),                                        # ellipsis_count
            len(re.findall(r"[!?]{2,}", t)),                       # repeated_punctuation
            len(words),                                            # word_count
            t.count("\n"),                                         # paragraph_count
        ]
        rows.append(row)

    return np.array(rows, dtype=np.float32)


# ── Feature Preparation ───────────────────────────────────────────────────────

def prepare_features(df, tfidf_max_features: int = 15000, test_size: float = 0.2):
    """
    Build full feature matrix: TF-IDF + handcrafted features combined.

    Args:
        df                 : DataFrame with columns 'content' and 'label'
        tfidf_max_features : number of TF-IDF vocab features (default 15000)
        test_size          : fraction for test split (default 0.2)

    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    X = df["content"]
    y = df["label"]

    # ── Balance check before split ─────────────────────────────────────────────
    fake_count = (y == 0).sum()
    real_count = (y == 1).sum()
    ratio      = min(fake_count, real_count) / max(fake_count, real_count)

    if ratio < 0.5:
        print(f"[train] WARNING: Class imbalance detected (Fake={fake_count}, Real={real_count}, ratio={ratio:.2f})")
        print(f"[train] Balancing by undersampling majority class...")

        fake_df = df[df["label"] == 0]
        real_df = df[df["label"] == 1]
        min_size = min(len(fake_df), len(real_df))

        fake_df = fake_df.sample(min_size, random_state=42)
        real_df = real_df.sample(min_size, random_state=42)
        df      = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42)

        X = df["content"]
        y = df["label"]
        print(f"[train] After balancing: {len(df)} total ({min_size} fake, {min_size} real)")

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = 42,
        stratify     = y
    )

    print(f"[train] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("[train] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words   = "english",
        ngram_range  = (1, 2),
        max_features = tfidf_max_features,
        sublinear_tf = True,    # log(1+tf) — normalizes long article bias
        min_df       = 2,       # ignore terms appearing in <2 docs (noise)
        max_df       = 0.95     # ignore terms in >95% of docs (too common)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ── Handcrafted features ──────────────────────────────────────────────────
    print("[train] Extracting handcrafted features...")
    X_train_hc = sp.csr_matrix(extract_handcrafted(X_train))
    X_test_hc  = sp.csr_matrix(extract_handcrafted(X_test))

    # ── Combine ───────────────────────────────────────────────────────────────
    X_train_full = sp.hstack([X_train_tfidf, X_train_hc])
    X_test_full  = sp.hstack([X_test_tfidf,  X_test_hc])

    print(f"[train] Feature shape: {X_train_full.shape}")
    print(f"        ({tfidf_max_features} TF-IDF + {len(FEATURE_NAMES)} handcrafted)")

    return X_train_full, X_test_full, y_train, y_test, vectorizer