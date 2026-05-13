# Regression using Two Models
# Dataset created using make_regression()

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------
# Create Regression Dataset
# ----------------------------------

X, y = make_regression(
    n_samples=100,
    n_features=8,
    noise=15,
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
# 1. Linear Regression
# ----------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

# ----------------------------------
# 2. Decision Tree Regression
# ----------------------------------

tree_model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

# ----------------------------------
# Evaluation
# ----------------------------------

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, linear_pred))
print("R2 Score:", r2_score(y_test, linear_pred))

print("\nDecision Tree Regression")
print("MSE:", mean_squared_error(y_test, tree_pred))
print("R2 Score:", r2_score(y_test, tree_pred))

# ----------------------------------
# Plot 1: Actual vs Predicted
# Linear Regression
# ----------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(y_test, linear_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Linear Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.show()

# ----------------------------------
# Plot 2: Actual vs Predicted
# Decision Tree Regression
# ----------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(y_test, tree_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.title("Decision Tree Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.show()