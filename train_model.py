import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data   = []
labels = []

# Read all CSV files from dataset folder
for file in os.listdir("dataset"):
    if file.endswith(".csv"):
        gesture_name = file.replace(".csv", "")
        df = pd.read_csv(f"dataset/{file}", header=None)
        data.append(df)
        labels += [gesture_name] * len(df)
        print(f"  Loaded '{gesture_name}' — {len(df)} samples")

print(f"\nTotal gestures : {len(set(labels))}")
print(f"Total samples  : {len(labels)}")

X = pd.concat(data, ignore_index=True)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# probability=True  →  enables model.predict_proba()
#                       which powers the live confidence bar
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n── Evaluation ──────────────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ── Confusion matrix plot ────────────────────────────────────────────────────
labels_sorted = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[l.replace("_", " ") for l in labels_sorted],
            yticklabels=[l.replace("_", " ") for l in labels_sorted])
plt.title("Confusion Matrix", fontsize=14)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()

if not os.path.exists("model"):
    os.makedirs("model")

plt.savefig("model/confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved → model/confusion_matrix.png")

# ── Save model ───────────────────────────────────────────────────────────────
pickle.dump(model, open("model/gesture_model.pkl", "wb"))
print("Model saved → model/gesture_model.pkl")