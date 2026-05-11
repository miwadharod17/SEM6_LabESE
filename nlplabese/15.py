import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Chunk rule
grammar = "NP: {<DT><JJ><NN>}"

parser = nltk.RegexpParser(grammar)

# User input
text = input("Enter a sentence: ")

# Tokenize
words = nltk.word_tokenize(text)

# POS tagging
tagged = nltk.pos_tag(words)

# Parse
result = parser.parse(tagged)

print(result)