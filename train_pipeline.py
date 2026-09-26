"""
train_pipeline.py
-----------------
Usage: python train_pipeline.py  (from project root)

Datasets needed in data/ folder:
    Fake.csv + True.csv   → already have  (Kaggle)
    IFND.csv              → already have  (Indian claims)
    data.csv              → already have  (COVID)
    indian_news.csv       → kaggle.com/datasets/imbikramsaha/fake-real-news
                            Download → rename to indian_news.csv
    liar_train.csv        → https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
                            Extract → rename train.tsv to liar_train.csv

MPD dataset (cryptexcode) is unavailable on Kaggle — replaced by indian_news.csv
"""

import sys, os

SRC_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, SRC_DIR)

from utils       import load_and_clean
from train       import prepare_features
from train_model import train_and_evaluate


def main():
    print("=" * 55)
    print("  FAKE NEWS DETECTOR — TRAINING PIPELINE")
    print("=" * 55)

    print("\n[1/3] Loading and combining datasets...")
    df = load_and_clean(
        fake_path         = os.path.join(DATA_DIR, "Fake.csv"),
        true_path         = os.path.join(DATA_DIR, "True.csv"),
        ifnd_path         = os.path.join(DATA_DIR, "IFND.csv"),
        covid_path        = os.path.join(DATA_DIR, "data.csv"),
        indian_news_path  = os.path.join(DATA_DIR, "indian_news.csv"),
        liar_path         = os.path.join(DATA_DIR, "liar_train.csv"),
        # bharat_path     = os.path.join(DATA_DIR, "bharat_news.csv"),  # optional
    )

    print("\n[2/3] Preparing features (TF-IDF + handcrafted)...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(
        df, tfidf_max_features=20000, test_size=0.2
    )

    print("\n[3/3] Training models (XGBoost + RF + Ensemble)...")
    train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer)

    print("\n✅ Training complete. Models saved to models/")
    print("   Run pipeline.py to test predictions.")


if __name__ == "__main__":
    main()