import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Elon Musk is the CEO of Tesla in California."

# Process text
doc = nlp(text)

# Extract named entities
for ent in doc.ents:
    print(ent.text, "-", ent.label_)