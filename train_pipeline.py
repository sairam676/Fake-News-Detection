"""
train_pipeline.py
-----------------
Run this once to train all models on combined dataset.

Usage:
    python train_pipeline.py   (run from project root)

Downloads needed (place in data/ folder):
    Fake.csv + True.csv  → kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    IFND.csv             → kaggle.com/datasets/sonalgarg174/ifnd-dataset
    data.csv             → kaggle.com/datasets/thesumitbanik/covid-fake-news-dataset
    isot_fake.csv        → kaggle.com/datasets/emineyetm/fake-news-detection-datasets
    isot_true.csv          (download both files, rename to isot_fake.csv / isot_true.csv)

Why ISOT helps
--------------
The Kaggle dataset is heavily US-election focused.  ISOT's real-news articles
come from Reuters (broad world coverage) so TF-IDF learns more neutral real-news
patterns.  Indian government/policy articles no longer look "foreign" to the model
because Reuters covered many of the same topics (demonetization, ISRO, etc.).
"""

import sys
import os

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
        fake_path      = os.path.join(DATA_DIR, "Fake.csv"),
        true_path      = os.path.join(DATA_DIR, "True.csv"),
        ifnd_path      = os.path.join(DATA_DIR, "IFND.csv"),
        covid_path     = os.path.join(DATA_DIR, "data.csv"),
        isot_fake_path = os.path.join(DATA_DIR, "isot_fake.csv"),
        isot_true_path = os.path.join(DATA_DIR, "isot_true.csv"),
    )

    print("\n[2/3] Preparing features (TF-IDF + handcrafted)...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(
        df,
        tfidf_max_features=15000,
        test_size=0.2
    )

    print("\n[3/3] Training models (XGBoost + RF + Ensemble)...")
    train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer)

    print("\n✅ Training complete. Models saved to models/")
    print("   Run pipeline.py to test predictions.")


if __name__ == "__main__":
    main()