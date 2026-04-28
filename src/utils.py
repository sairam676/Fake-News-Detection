"""
utils.py
--------
Loads and combines four datasets:

    1. Kaggle Fake/True  → Fake.csv + True.csv
    2. IFND Indian       → IFND.csv
    3. COVID Fake News   → data.csv
    4. ISOT              → True.csv + Fake.csv  (University of Victoria)
                          kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Place all files in data/ folder.

KEY FIX: Each dataset is balanced before merging.
COVID was 9727 fake : 474 real (20:1) — was poisoning the combined dataset.

ISOT: ~44k articles, well-balanced, Reuters real / GossipCop+PolitiFact fake.
Adds diversity that dilutes US-election TF-IDF bias from the Kaggle dataset.
Place as data/isot_fake.csv and data/isot_true.csv  (rename if needed).
"""

import pandas as pd
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _read_csv_safe(path):
    """Try multiple encodings until one works."""
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise ValueError(f"Could not read {path} with any encoding.")


def _balance(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Undersample majority class so fake:real ratio is at most 2:1.
    COVID dataset was 20:1 — this was the main cause of LR collapse to 58%.
    """
    fake = df[df["label"] == 0]
    real = df[df["label"] == 1]
    ratio = min(len(fake), len(real)) / max(len(fake), len(real))

    if ratio < 0.5:
        n    = min(len(fake), len(real))
        fake = fake.sample(n, random_state=42)
        real = real.sample(n, random_state=42)
        df   = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[utils] {name} balanced → {n} fake + {n} real = {len(df)} total")
    else:
        print(f"[utils] {name} OK (ratio={ratio:.2f}) → Fake={len(fake)}, Real={len(real)}")

    return df


def _load_kaggle(fake_path, true_path) -> pd.DataFrame:
    fake_df = _read_csv_safe(fake_path)
    true_df = _read_csv_safe(true_path)

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df[['title', 'text', 'label']].dropna()
    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

    print(f"[utils] Kaggle loaded    : {len(df)} articles")
    return _balance(df[['content', 'label']], "Kaggle")


def _load_isot(fake_path, true_path) -> pd.DataFrame:
    """
    ISOT dataset — University of Victoria.
    Real news: Reuters articles.  Fake news: GossipCop + PolitiFact.
    Expected columns: title, text, subject, date  (same schema as Kaggle dataset).
    Download: kaggle.com/datasets/emineyetm/fake-news-detection-datasets
    Rename downloaded files to isot_fake.csv and isot_true.csv.

    Why this helps: Reuters articles cover world events broadly, not just
    US elections, which dilutes the TF-IDF political-keyword bias that causes
    Indian government/policy articles to be misclassified.
    """
    fake_df = _read_csv_safe(fake_path)
    true_df = _read_csv_safe(true_path)

    # Normalise column names — ISOT sometimes ships with trailing spaces
    fake_df.columns = fake_df.columns.str.strip()
    true_df.columns = true_df.columns.str.strip()

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Accept title+text or just text
    if 'title' in df.columns and 'text' in df.columns:
        df = df[['title', 'text', 'label']].dropna()
        df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    elif 'text' in df.columns:
        df = df[['text', 'label']].dropna()
        df = df.rename(columns={'text': 'content'})
    else:
        print("[utils] ISOT: could not find text column, skipping")
        return pd.DataFrame(columns=['content', 'label'])

    print(f"[utils] ISOT loaded      : {len(df)} articles")
    return _balance(df[['content', 'label']], "ISOT")


def _load_ifnd(ifnd_path) -> pd.DataFrame:
    df = _read_csv_safe(ifnd_path)
    df.columns = df.columns.str.strip()

    print(f"[utils] IFND columns: {list(df.columns)}")

    if 'Statement' not in df.columns or 'Label' not in df.columns:
        print("[utils] IFND: required columns not found, skipping")
        return pd.DataFrame(columns=['content', 'label'])

    df = df[['Statement', 'Label']].dropna()
    df = df.rename(columns={'Statement': 'content', 'Label': 'label'})

    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({
        'fake': 0, '0': 0, 'false': 0, 'no': 0,
        'real': 1, '1': 1, 'true': 1,  'yes': 1
    })
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"[utils] IFND loaded      : {len(df)} articles "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "IFND")


def _load_covid(covid_path) -> pd.DataFrame:
    df = _read_csv_safe(covid_path)
    df.columns = df.columns.str.strip()

    print(f"[utils] COVID columns: {list(df.columns)}")

    if 'headlines' not in df.columns or 'outcome' not in df.columns:
        print("[utils] COVID: required columns not found, skipping")
        return pd.DataFrame(columns=['content', 'label'])

    df = df[['headlines', 'outcome']].dropna()
    df = df.rename(columns={'headlines': 'content', 'outcome': 'label'})

    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({
        'fake': 0, '0': 0, 'false': 0,
        'real': 1, '1': 1, 'true': 1
    })
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"[utils] COVID loaded     : {len(df)} articles "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "COVID")   # ← fixes the 9727:474 problem


def load_and_clean(
    fake_path=None,
    true_path=None,
    ifnd_path=None,
    covid_path=None,
    isot_fake_path=None,
    isot_true_path=None,
) -> pd.DataFrame:
    dfs = []

    # ── Kaggle ────────────────────────────────────────────────────────────────
    fake_p = fake_path or os.path.join(DATA_DIR, "Fake.csv")
    true_p = true_path or os.path.join(DATA_DIR, "True.csv")
    if os.path.exists(fake_p) and os.path.exists(true_p):
        dfs.append(_load_kaggle(fake_p, true_p))
    else:
        print("[utils] Kaggle dataset not found — skipping")

    # ── ISOT ──────────────────────────────────────────────────────────────────
    isot_fake_p = isot_fake_path or os.path.join(DATA_DIR, "isot_fake.csv")
    isot_true_p = isot_true_path or os.path.join(DATA_DIR, "isot_true.csv")
    if os.path.exists(isot_fake_p) and os.path.exists(isot_true_p):
        d = _load_isot(isot_fake_p, isot_true_p)
        if len(d) > 0:
            dfs.append(d)
    else:
        print("[utils] ISOT dataset not found — skipping "
              "(download from kaggle.com/datasets/emineyetm/fake-news-detection-datasets "
              "and rename to isot_fake.csv / isot_true.csv)")

    # ── IFND ──────────────────────────────────────────────────────────────────
    ifnd_p = ifnd_path or os.path.join(DATA_DIR, "IFND.csv")
    if not os.path.exists(ifnd_p):
        ifnd_p = os.path.join(DATA_DIR, "ifnd.csv")
    if os.path.exists(ifnd_p):
        d = _load_ifnd(ifnd_p)
        if len(d) > 0:
            dfs.append(d)
    else:
        print("[utils] IFND.csv not found — skipping")

    # ── COVID ─────────────────────────────────────────────────────────────────
    covid_p = covid_path or os.path.join(DATA_DIR, "data.csv")
    if os.path.exists(covid_p):
        d = _load_covid(covid_p)
        if len(d) > 0:
            dfs.append(d)
    else:
        print("[utils] data.csv not found — skipping")

    if not dfs:
        raise FileNotFoundError(
            "No datasets found. Place Fake.csv + True.csv in data/ folder."
        )

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['content', 'label'])
    combined = combined[combined['content'].str.len() > 50]
    combined = combined.drop_duplicates(subset=['content'])
    combined['label'] = combined['label'].astype(int)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[utils] Combined dataset : {len(combined)} articles")
    print(f"[utils] Fake             : {(combined['label']==0).sum()}")
    print(f"[utils] Real             : {(combined['label']==1).sum()}")
    print(f"[utils] Datasets loaded  : {len(dfs)}\n")

    return combined[['content', 'label']]


if __name__ == "__main__":
    df = load_and_clean()
    print(df['label'].value_counts())