from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer):
    
    model = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        class_weight='balanced'
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model + vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModel saved in /models")

    return model
