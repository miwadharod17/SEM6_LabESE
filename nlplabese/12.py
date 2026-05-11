import random

# Training text
text = "I love python and I love coding and python is easy"

# Split into words
words = text.split()

# Bigram count dictionary
bigram_count = {}

for i in range(len(words) - 1):
    word1 = words[i]
    word2 = words[i + 1]

    if word1 not in bigram_count:
        bigram_count[word1] = {}

    if word2 not in bigram_count[word1]:
        bigram_count[word1][word2] = 0

    bigram_count[word1][word2] += 1

# Convert counts to probabilities
bigram_prob = {}

for word1 in bigram_count:
    total = sum(bigram_count[word1].values())
    bigram_prob[word1] = {}

    for word2 in bigram_count[word1]:
        bigram_prob[word1][word2] = bigram_count[word1][word2] / total

# Print probabilities
print("Bigram Probabilities:")
for word1 in bigram_prob:
    print(word1, "->", bigram_prob[word1])

# Generate text
current_word = random.choice(words)
generated = current_word

for _ in range(10):
    if current_word in bigram_prob:
        next_words = list(bigram_prob[current_word].keys())
        probabilities = list(bigram_prob[current_word].values())

        next_word = random.choices(next_words, probabilities)[0]

        generated += " " + next_word
        current_word = next_word
    else:
        break

print("\nGenerated Text:")
print(generated)