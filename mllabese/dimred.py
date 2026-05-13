# Dimensionality Reduction using PCA and SVD
# Dataset created using make_classification()

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD

# ----------------------------------
# Create Dataset with Many Features
# ----------------------------------

X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

# Convert to DataFrame
columns = [f"Feature_{i}" for i in range(1, 11)]

df = pd.DataFrame(X, columns=columns)

# Add target column
df["Target"] = y

print("Original Dataset:\n")
print(df.head())

# ----------------------------------
# Separate Features
# ----------------------------------

X = df.drop("Target", axis=1)

# ----------------------------------
# 1. PCA
# ----------------------------------

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X)

pca_df = pd.DataFrame(
    pca_result,
    columns=["PC1", "PC2"]
)

print("\nPCA Reduced Data:\n")
print(pca_df.head())

# ----------------------------------
# PCA Scatter Plot
# ----------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=y
)

plt.title("PCA - 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()

# ----------------------------------
# PCA Explained Variance Plot
# ----------------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_
)

plt.title("PCA Explained Variance Ratio")
plt.ylabel("Variance Explained")

plt.show()

# ----------------------------------
# 2. SVD
# ----------------------------------

svd = TruncatedSVD(n_components=2)

svd_result = svd.fit_transform(X)

svd_df = pd.DataFrame(
    svd_result,
    columns=["SVD1", "SVD2"]
)

print("\nSVD Reduced Data:\n")
print(svd_df.head())

# ----------------------------------
# SVD Scatter Plot
# ----------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(
    svd_df["SVD1"],
    svd_df["SVD2"],
    c=y
)

plt.title("SVD - 2D Projection")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")

plt.show()

# ----------------------------------
# SVD Explained Variance Plot
# ----------------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ["SVD1", "SVD2"],
    svd.explained_variance_ratio_
)

plt.title("SVD Explained Variance Ratio")
plt.ylabel("Variance Explained")

plt.show()