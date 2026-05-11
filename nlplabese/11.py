# Hidden Markov Model - Brute Force

states = ["Rainy", "Sunny"]

start_prob = {
    "Rainy": 0.6,
    "Sunny": 0.4
}

transition = {
    "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
    "Sunny": {"Rainy": 0.4, "Sunny": 0.6}
}

emission = {
    "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
    "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1}
}

observations = ["walk", "shop", "clean"]

best_path = None
best_prob = 0

# Check all 8 possible paths
for s1 in states:
    for s2 in states:
        for s3 in states:

            path = [s1, s2, s3]

            prob = start_prob[s1]
            prob *= emission[s1][observations[0]]

            prob *= transition[s1][s2]
            prob *= emission[s2][observations[1]]

            prob *= transition[s2][s3]
            prob *= emission[s3][observations[2]]

            print("Path:", path)
            print("Probability:", round(prob, 6))
            print()

            if prob > best_prob:
                best_prob = prob
                best_path = path

print("Most Likely Path:", best_path)
print("Maximum Probability:", round(best_prob, 6))