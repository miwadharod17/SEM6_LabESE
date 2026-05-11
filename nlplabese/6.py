# Ambiguity Detection

# Input sentence
sentence = "The chicken is ready to eat"

# Convert sentence into words
words = sentence.lower().split()

# Default
ambiguity = "No Ambiguity"

# Simple rules

# Syntactic ambiguity:
# caused by prepositions
prepositions = ["with", "by", "on", "in"]

for word in words:
    if word in prepositions:
        ambiguity = "Syntactic Ambiguity"

# Semantic ambiguity:
# words having multiple meanings
semantic_words = ["bat", "bank", "light", "chicken"]

for word in words:
    if word in semantic_words:
        ambiguity = "Semantic Ambiguity"

# Output
print("Sentence:")
print(sentence)

print("\nAmbiguity Type:")
print(ambiguity)