# Cosine Similarity Calculation

import math

# Two vectors
A = [1, 2, 3]
B = [2, 1, 3]

# Step 1: Dot Product
dot_product = 0

for i in range(len(A)):
    dot_product += A[i] * B[i]

# Step 2: Magnitude of A
magnitude_A = 0

for value in A:
    magnitude_A += value ** 2

magnitude_A = math.sqrt(magnitude_A)

# Step 3: Magnitude of B
magnitude_B = 0

for value in B:
    magnitude_B += value ** 2

magnitude_B = math.sqrt(magnitude_B)

# Step 4: Cosine Similarity
cosine_similarity = dot_product / (magnitude_A * magnitude_B)

# Output
print("Dot Product:", dot_product)
print("Magnitude of A:", round(magnitude_A, 2))
print("Magnitude of B:", round(magnitude_B, 2))
print("Cosine Similarity:", round(cosine_similarity, 2))