# Simple Rule-Based Stemmer

# List of words
words = ["playing", "jumped", "walking", "cleaned", "reading"]

# Function for stemming
def stem_word(word):

    # Remove "ing"
    if word.endswith("ing"):
        return word[:-3]

    # Remove "ed"
    elif word.endswith("ed"):
        return word[:-2]

    # Return original word if no suffix found
    else:
        return word

# Apply stemming
stemmed_words = []

for word in words:
    stemmed_words.append(stem_word(word))

# Output
print("Original Words:")
print(words)

print("\nStemmed Words:")
print(stemmed_words)