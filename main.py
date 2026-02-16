from src.utils import load_and_clean
from src.train import prepare_features
from src.train_model import train_and_evaluate

def main():
    print("Loading and cleaning data...")
    df = load_and_clean("data/Fake.csv", "data/True.csv")

    print("Preparing features...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(df)

    print("Training model...")
    model = train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer)

if __name__ == "__main__":
    main()
