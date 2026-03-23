import pandas as pd
import matplotlib.pyplot as plt
import os

# Get project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fake_path = os.path.join(BASE_DIR, "data", "Fake.csv")
true_path = os.path.join(BASE_DIR, "data", "True.csv")

# Load datasets
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Add labels
fake["label"] = "Fake"
true["label"] = "Real"

# Combine datasets
df = pd.concat([fake, true])

# Count fake and real articles per subject
distribution = df.groupby(["subject", "label"]).size().unstack()

# Plot
distribution.plot(kind="bar", figsize=(10,5))

plt.title("Fake vs Real News Distribution by Class")
plt.xlabel("Classes (Subject Categories)")
plt.ylabel("Number of Articles")
plt.xticks(rotation=45)

plt.show()