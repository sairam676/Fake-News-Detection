import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Logistic Regression", "Naive Bayes"]

# Metrics
accuracy = [0.986, 0.954]
precision = [0.99, 0.96]
recall = [0.98, 0.95]
f1_score = [0.99, 0.95]

# X positions
x = np.arange(len(models))
width = 0.2

# Plot bars
plt.figure(figsize=(10,5))

plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1_score, width, label="F1-Score")

# Labels
plt.ylabel("Score")
plt.title("Baseline Model Performance Comparison (Logistic Regression vs Naive Bayes)")
plt.xticks(x, models)

# Add value labels on top
for i, v in enumerate(accuracy):
    plt.text(i - 1.5*width, v + 0.01, str(v), ha='center')

for i, v in enumerate(precision):
    plt.text(i - 0.5*width, v + 0.01, str(v), ha='center')

for i, v in enumerate(recall):
    plt.text(i + 0.5*width, v + 0.01, str(v), ha='center')

for i, v in enumerate(f1_score):
    plt.text(i + 1.5*width, v + 0.01, str(v), ha='center')

plt.legend()
plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()