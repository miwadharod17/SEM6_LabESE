# Naive Bayes Classification with Priors

# Prior probabilities
P_spam = 0.6
P_ham = 0.4

# Word probabilities for Spam
spam_probs = {
    "offer": 0.5,
    "win": 0.4,
    "money": 0.3
}

# Word probabilities for Ham
ham_probs = {
    "offer": 0.1,
    "win": 0.05,
    "money": 0.02
}

# Test sentence
sentence = "win money"

# Split into words
words = sentence.split()

# Start with priors
spam_score = P_spam
ham_score = P_ham

# Apply Naive Bayes multiplication
for word in words:

    if word in spam_probs:
        spam_score *= spam_probs[word]

    if word in ham_probs:
        ham_score *= ham_probs[word]

# Output
print("Spam Score:", spam_score)
print("Ham Score:", ham_score)

# Classification
if spam_score > ham_score:
    print("\nClassified as: Spam")
else:
    print("\nClassified as: Ham")