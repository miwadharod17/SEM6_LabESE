# pca.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------- DATASET ----------------

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, np.nan, 9],
    
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    
    "AssignmentScore": [20, 25, 30, 40, 50, 60, 70, 80, 85, 95],
    
    "SleepHours": [9, 8, 8, 7, 7, 6, 6, 5, 5, 4],
    
    "ProjectsCompleted": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
}

df = pd.DataFrame(data)

print("\nOriginal Dataset:\n")
print(df)

# ---------------- PREPROCESSING ----------------

# Check missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Handle missing values
imputer = SimpleImputer(strategy='mean')

df["StudyHours"] = imputer.fit_transform(df[["StudyHours"]])

print("\nAfter Preprocessing:\n")
print(df)

# ---------------- FEATURES ----------------

X = df

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------- PCA ----------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# ---------------- PCA OUTPUT ----------------

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

print("\nPCA Transformed Data:\n")
print(pca_df)

# ---------------- EXPLAINED VARIANCE ----------------

print("\nExplained Variance Ratio:\n")
print(pca.explained_variance_ratio_)

# ---------------- VISUALIZATION ----------------

plt.figure(figsize=(7,5))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    s=100
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA Dimensionality Reduction")

plt.show()