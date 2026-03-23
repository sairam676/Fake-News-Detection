from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def train_and_evaluate(X_train, X_test, y_train, y_test, vectorizer):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB()
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():

        print("\n==============================")
        print(f"Training {name}")
        print("==============================")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("\nAccuracy:", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Save models separately
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, f"models/{filename}")

    # save vectorizer once
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModels saved in /models")