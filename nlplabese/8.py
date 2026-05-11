# TF (Term Frequency) Calculation

# Input document
document = "python is easy and python is powerful"

# Convert document into words
words = document.lower().split()

# Total number of words
total_words = len(words)

# Find unique words
unique_words = []

for word in words:
    if word not in unique_words:
        unique_words.append(word)

# Calculate TF
print("Word\tTF")

for word in unique_words:

    count = words.count(word)

    tf = count / total_words

    print(word, "\t", round(tf, 2))
    