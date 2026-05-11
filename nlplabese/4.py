# Morphological Analysis

# Dictionary containing word breakdown
words = {
    "unhappiness": ["un", "happy", "ness"],
    "replayed": ["re", "play", "ed"],
    "international": ["inter", "nation", "al"]
}

# Lists of known bound morphemes
bound_morphemes = ["un", "ness", "re", "ed", "inter", "al"]

# Analyze words
for word, morphemes in words.items():

    print("\nWord:", word)

    for morpheme in morphemes:

        if morpheme in bound_morphemes:
            morpheme_type = "Bound Morpheme"
        else:
            morpheme_type = "Free Morpheme"

        print(morpheme, "->", morpheme_type)