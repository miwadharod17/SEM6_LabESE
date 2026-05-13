# Classification using Two Models
# Dataset created using make_classification()

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------
# Create Classification Dataset
# ----------------------------------

X, y = make_classification(
    n_samples=200,
    n_features=8,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Convert to DataFrame
columns = [f"Feature_{i}" for i in range(1, 9)]

df = pd.DataFrame(X, columns=columns)

df["Target"] = y

print("Dataset:\n")
print(df.head())

# ----------------------------------
# Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# 1. Logistic Regression
# ----------------------------------

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

# ----------------------------------
# 2. Decision Tree Classifier
# ----------------------------------

tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

# ----------------------------------
# Accuracy Scores
# ----------------------------------

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, log_pred))

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, tree_pred))

# ----------------------------------
# Confusion Matrix - Logistic Regression
# ----------------------------------

log_cm = confusion_matrix(y_test, log_pred)

plt.figure(figsize=(5, 4))

plt.imshow(log_cm)

plt.title("Logistic Regression Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ----------------------------------
# Confusion Matrix - Decision Tree
# ----------------------------------

tree_cm = confusion_matrix(y_test, tree_pred)

plt.figure(figsize=(5, 4))

plt.imshow(tree_cm)

plt.title("Decision Tree Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()