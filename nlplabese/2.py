# Stopword Removal without using NLP libraries

# Input sentence
sentence = "This is a simple example of stopword removal in Python"

# Manual list of stopwords
stopwords = ["is", "a", "of", "in", "the", "and", "to"]

# Split sentence into words
words = sentence.split()

# Remove stopwords
cleaned_words = []

for word in words:
    if word.lower() not in stopwords:
        cleaned_words.append(word)

# Join words back into sentence
cleaned_text = " ".join(cleaned_words)

# Output
print("Original Sentence:")
print(sentence)

print("\nCleaned Text:")
print(cleaned_text)