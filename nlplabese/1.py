# Tokenization Task using split(".")

# Input paragraph
paragraph = "Natural Language Processing is interesting. It helps computers understand language. Tokenization is the first step."

# Step 1: Split into sentences
sentences = paragraph.split(".")

# Remove empty strings and extra spaces
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

# Step 2: Split sentences into words
words = []

for sentence in sentences:
    sentence_words = sentence.split()
    words.extend(sentence_words)

# Step 3: Count total tokens
total_tokens = len(words)

# Output
print("Sentences:")
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

print("\nWords:")
print(words)

print("\nTotal Tokens:", total_tokens)