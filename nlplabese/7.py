# Bag of Words Implementation

# Input sentences
sentences = [
    "I love python",
    "python is easy",
    "I love coding"
]

# Step 1: Create vocabulary manually
vocabulary = []

for sentence in sentences:
    words = sentence.lower().split()

    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)

# Step 2: Create Bag-of-Words matrix
bow_matrix = []

for sentence in sentences:

    words = sentence.lower().split()

    row = []

    for vocab_word in vocabulary:
        count = words.count(vocab_word)
        row.append(count)

    bow_matrix.append(row)

# Output
print("Vocabulary:")
print(vocabulary)

print("\nBag of Words Matrix:")

for row in bow_matrix:
    print(row)