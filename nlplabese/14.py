import re

# Input text
text = "Hello World! Python 2026 is Awesome, Score = 95."

print("Original Text:", text)

# Remove punctuation
text = re.sub(r'[^\w\s]', '', text)   

# Remove numbers
text = re.sub(r'\d+', '', text)

# Convert to lowercase
text = text.lower()

print("Cleaned Text:", text)