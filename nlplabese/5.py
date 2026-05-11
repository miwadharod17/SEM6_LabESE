# Simple Rule-Based POS Tagging

# Input sentence
sentence = "Rahul quickly completed the assigned work"

# Split sentence into words
words = sentence.split()

# POS tagging using simple rules
for word in words:

    # Convert to lowercase for checking
    w = word.lower()

    if w.endswith("ly"):
        tag = "Adverb"

    elif w.endswith("ing"):
        tag = "Verb"

    elif w.endswith("ed"):
        tag = "Verb"

    elif w in ["a", "an", "the"]:
        tag = "Article"

    elif w in ["is", "am", "are", "was", "were"]:
        tag = "Verb"

    else:
        tag = "Noun"

    print(word, "->", tag)