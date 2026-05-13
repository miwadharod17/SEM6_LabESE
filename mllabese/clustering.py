# kmeans_multifeature.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---------------- DATASET ----------------

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, np.nan, 9],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "AssignmentScore": [20, 25, 30, 40, 50, 60, 70, 80, 85, 95],
    "SleepHours": [9, 8, 8, 7, 7, 6, 6, 5, 5, 4]
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

X = df[[
    "StudyHours",
    "Attendance",
    "AssignmentScore",
    "SleepHours"
]]

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------- ELBOW METHOD ----------------

wcss = []

for i in range(1, 6):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(6,4))

plt.plot(range(1, 6), wcss, marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()

# ---------------- K-MEANS MODEL ----------------

model = KMeans(
    n_clusters=2,
    random_state=42
)

# Train model
model.fit(X_scaled)

# Predict clusters
clusters = model.predict(X_scaled)

# Add Cluster Column
df["Cluster"] = clusters

print("\nClustered Dataset:\n")
print(df)

# ---------------- CENTROIDS ----------------

print("\nCentroids:\n")
print(model.cluster_centers_)

# ---------------- VISUALIZATION ----------------

# Since we have 4 features,
# visualize only 2 features

plt.figure(figsize=(7,5))

plt.scatter(
    df["StudyHours"],
    df["AssignmentScore"],
    c=df["Cluster"],
    s=100
)

# Convert centroids back
centroids = scaler.inverse_transform(model.cluster_centers_)

plt.scatter(
    centroids[:,0],   # StudyHours centroid
    centroids[:,2],   # AssignmentScore centroid
    marker='X',
    s=300
)

plt.xlabel("Study Hours")
plt.ylabel("Assignment Score")

plt.title("Student Clustering")

plt.show()