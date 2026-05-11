# Reference and candidate sentences
reference = "the cat is on the mat"
candidate = "the cat is on mat"

# Split into words
ref_words = reference.split()
cand_words = candidate.split()

# Count matching words
match = 0

for word in cand_words:
    if word in ref_words:
        match += 1

# Calculate BLEU score
bleu_score = match / len(cand_words)

print("Reference Sentence:", reference)
print("Candidate Sentence:", candidate)
print("BLEU Score:", round(bleu_score, 2))