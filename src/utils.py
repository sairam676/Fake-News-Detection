"""
utils.py
--------
Loads and combines datasets for fake news detection.

Datasets:
    1. Kaggle Fake/True     → Fake.csv + True.csv        (US politics, 44k)
    2. IFND Indian          → IFND.csv                   (Indian claims, 37k)
    3. COVID Fake News      → data.csv                   (health, balanced)
    4. Indian Fake News     → indian_news.csv             (imbikramsaha, ~10k)
       kaggle.com/datasets/imbikramsaha/fake-real-news
       Download → rename to indian_news.csv → place in data/
    5. LIAR dataset         → liar_train.csv              (political claims, 6-class → binary)
       https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
       Extract → rename train.tsv to liar_train.csv → place in data/

MPD dataset (cryptexcode) is unavailable on Kaggle — replaced by indian_news.csv
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RANDOM_STATE = 42


def _read_csv_safe(path, sep=','):
    for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(path, encoding=enc, sep=sep)
        except Exception:
            continue
    raise ValueError(f"Could not read {path} with any encoding.")


def _balance(df, name):
    fake  = df[df["label"] == 0]
    real  = df[df["label"] == 1]
    ratio = min(len(fake), len(real)) / max(len(fake), len(real))
    if ratio < 0.5:
        n    = min(len(fake), len(real))
        fake = fake.sample(n, random_state=RANDOM_STATE)
        real = real.sample(n, random_state=RANDOM_STATE)
        df   = pd.concat([fake, real]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[utils] {name} balanced → {n} fake + {n} real = {len(df)} total")
    else:
        print(f"[utils] {name} OK (ratio={ratio:.2f}) → Fake={len(fake)}, Real={len(real)}")
    return df


def _load_kaggle(fake_path, true_path):
    fake_df = _read_csv_safe(fake_path)
    true_df = _read_csv_safe(true_path)
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df[['title', 'text', 'label']].dropna()
    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    print(f"[utils] Kaggle loaded    : {len(df)} articles")
    return _balance(df[['content', 'label']], "Kaggle")


def _load_ifnd(path):
    df = _read_csv_safe(path)
    df.columns = df.columns.str.strip()
    print(f"[utils] IFND columns: {list(df.columns)}")
    if 'Statement' not in df.columns or 'Label' not in df.columns:
        print("[utils] IFND: required columns not found, skipping")
        return pd.DataFrame(columns=['content', 'label'])
    df = df[['Statement', 'Label']].dropna()
    df = df.rename(columns={'Statement': 'content', 'Label': 'label'})
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({'fake': 0, '0': 0, 'false': 0, 'no': 0,
                                    'real': 1, '1': 1, 'true': 1, 'yes': 1})
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"[utils] IFND loaded      : {len(df)} articles "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "IFND")


def _load_covid(path):
    df = _read_csv_safe(path)
    df.columns = df.columns.str.strip()
    print(f"[utils] COVID columns: {list(df.columns)}")
    if 'headlines' not in df.columns or 'outcome' not in df.columns:
        print("[utils] COVID: required columns not found, skipping")
        return pd.DataFrame(columns=['content', 'label'])
    df = df[['headlines', 'outcome']].dropna()
    df = df.rename(columns={'headlines': 'content', 'outcome': 'label'})
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({'fake': 0, '0': 0, 'false': 0,
                                    'real': 1, '1': 1, 'true': 1})
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"[utils] COVID loaded     : {len(df)} articles "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "COVID")


def _load_indian_news(path):
    """
    Load imbikramsaha/fake-real-news Indian dataset.
    kaggle.com/datasets/imbikramsaha/fake-real-news
    Rename downloaded file to indian_news.csv and place in data/

    Auto-detects label column and text column.
    Label values: 'Fake'/'Real' strings or 0/1.
    """
    df = _read_csv_safe(path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"[utils] IndianNews columns: {list(df.columns)}")

    # Find label column
    label_col = next((c for c in ['label', 'class', 'type', 'category', 'fake'] if c in df.columns), None)
    if not label_col:
        print("[utils] IndianNews: cannot find label column, skipping")
        return pd.DataFrame(columns=['content', 'label'])

    # Find text column
    text_col = next((c for c in ['text', 'content', 'body', 'article', 'news', 'title', 'statement']
                     if c in df.columns), None)
    if not text_col:
        print(f"[utils] IndianNews: cannot find text column in {list(df.columns)}, skipping")
        return pd.DataFrame(columns=['content', 'label'])

    # Combine title + text if both exist
    if 'title' in df.columns and text_col != 'title':
        df['content'] = df['title'].astype(str) + " " + df[text_col].astype(str)
    else:
        df['content'] = df[text_col].astype(str)

    df = df[['content', label_col]].rename(columns={label_col: 'label'}).dropna()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({
        'fake': 0, '0': 0, 'false': 0, 'no': 0,
        'real': 1, '1': 1, 'true': 1, 'yes': 1
    })
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['content'].str.len() > 30]

    print(f"[utils] IndianNews loaded: {len(df)} articles "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "IndianNews")


def _load_bharat(path):
    """
    Load BharatFakeNewsKosh dataset (optional backup for Indian news).
    kaggle.com/datasets/man2191989/bharatfakenewskosh
    Download English CSV → rename to bharat_news.csv → place in data/
    """
    df = _read_csv_safe(path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"[utils] Bharat columns: {list(df.columns)}")

    label_col = next((c for c in ['label', 'verdict', 'class', 'type', 'fake', 'tag'] if c in df.columns), None)
    text_col  = next((c for c in ['news', 'text', 'content', 'body', 'article', 'statement', 'title']
                      if c in df.columns), None)

    if not label_col or not text_col:
        print(f"[utils] Bharat: cannot find required columns (label={label_col}, text={text_col}), skipping")
        return pd.DataFrame(columns=['content', 'label'])

    df = df[[text_col, label_col]].rename(columns={text_col: 'content', label_col: 'label'}).dropna()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({
        'fake': 0, '0': 0, 'false': 0, 'no': 0, 'misleading': 0,
        'real': 1, '1': 1, 'true': 1, 'yes': 1, 'valid': 1, 'verified': 1
    })
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['content'].str.len() > 30]

    print(f"[utils] Bharat loaded    : {len(df)} articles "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "Bharat")


def _load_liar(path):
    """
    Load LIAR dataset (political claims, 6-class → binary).
    https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    Extract → rename train.tsv to liar_train.csv → place in data/

    6-class labels collapsed:
        pants-fire / false / barely-true  → 0 (fake)
        half-true / mostly-true / true    → 1 (real)

    TSV format — no header row:
        col 0: ID, col 1: label, col 2: statement, col 3-13: metadata
    """
    # Try tab-separated first (original .tsv), then comma
    df = None
    for sep in ['\t', ',']:
        try:
            tmp = pd.read_csv(path, sep=sep, header=None, encoding='utf-8')
            if tmp.shape[1] >= 3:
                df = tmp
                break
        except Exception:
            continue

    if df is None:
        print("[utils] LIAR: could not parse file, skipping")
        return pd.DataFrame(columns=['content', 'label'])

    print(f"[utils] LIAR columns count: {df.shape[1]}, rows: {df.shape[0]}")

    # Standard LIAR TSV: col 1 = label, col 2 = statement
    df = df[[1, 2]].copy()
    df.columns = ['label', 'content']
    df = df.dropna()

    FAKE_LABELS = {'pants-fire', 'false', 'barely-true'}
    REAL_LABELS = {'half-true', 'mostly-true', 'true'}

    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].apply(
        lambda x: 0 if x in FAKE_LABELS else (1 if x in REAL_LABELS else None)
    )
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['content'].str.len() > 20]

    print(f"[utils] LIAR loaded      : {len(df)} claims "
          f"(Fake={(df['label']==0).sum()}, Real={(df['label']==1).sum()})")
    return _balance(df[['content', 'label']], "LIAR")


def load_and_clean(
    fake_path=None,
    true_path=None,
    ifnd_path=None,
    covid_path=None,
    isot_fake_path=None,   # kept for backward compat — ignored
    isot_true_path=None,
    mpd_fake_path=None,    # kept for backward compat — ignored (dataset unavailable)
    mpd_true_path=None,
    indian_news_path=None,
    bharat_path=None,
    liar_path=None,
) -> pd.DataFrame:
    dfs = []

    # Kaggle
    fake_p = fake_path or os.path.join(DATA_DIR, "Fake.csv")
    true_p = true_path or os.path.join(DATA_DIR, "True.csv")
    if os.path.exists(fake_p) and os.path.exists(true_p):
        dfs.append(_load_kaggle(fake_p, true_p))
    else:
        print("[utils] Kaggle not found — skipping")

    # IFND
    ifnd_p = ifnd_path or os.path.join(DATA_DIR, "IFND.csv")
    if not os.path.exists(ifnd_p):
        ifnd_p = os.path.join(DATA_DIR, "ifnd.csv")
    if os.path.exists(ifnd_p):
        d = _load_ifnd(ifnd_p)
        if len(d): dfs.append(d)
    else:
        print("[utils] IFND not found — skipping")

    # COVID
    covid_p = covid_path or os.path.join(DATA_DIR, "data.csv")
    if os.path.exists(covid_p):
        d = _load_covid(covid_p)
        if len(d): dfs.append(d)
    else:
        print("[utils] COVID not found — skipping")

    # Indian Fake News (MPD replacement)
    indian_p = indian_news_path or os.path.join(DATA_DIR, "indian_news.csv")
    if os.path.exists(indian_p):
        d = _load_indian_news(indian_p)
        if len(d): dfs.append(d)
    else:
        print("[utils] IndianNews not found — skipping")
        print("        Download: kaggle.com/datasets/imbikramsaha/fake-real-news")
        print("        Rename to indian_news.csv in data/")

    # BharatFakeNewsKosh (optional)
    bharat_p = bharat_path or os.path.join(DATA_DIR, "bharat_news.csv")
    if os.path.exists(bharat_p):
        d = _load_bharat(bharat_p)
        if len(d): dfs.append(d)
    # silently skip — optional dataset

    # LIAR
    liar_p = liar_path or os.path.join(DATA_DIR, "liar_train.csv")
    if not os.path.exists(liar_p):
        liar_p = os.path.join(DATA_DIR, "train.tsv")   # fallback original name
    if os.path.exists(liar_p):
        d = _load_liar(liar_p)
        if len(d): dfs.append(d)
    else:
        print("[utils] LIAR not found — skipping")
        print("        Download: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
        print("        Extract → rename train.tsv to liar_train.csv in data/")

    if not dfs:
        raise FileNotFoundError("No datasets found. Place Fake.csv + True.csv in data/")

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