# Simple RNN - Complete Experiment
# Includes:
# 1. Next word prediction
# 2. Effect of sequence length
# 3. Limitation of Simple RNN
# 4. Short vs Long sequence comparison

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# TEXT DATA
# ---------------------------------------------------

text = """
deep learning is interesting
deep learning is powerful
machine learning is useful
artificial intelligence is growing
artificial intelligence changes technology
deep neural networks learn patterns
machine learning improves prediction
"""

# ---------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Total Words:", total_words)
print("Word Index:\n", tokenizer.word_index)

# ---------------------------------------------------
# CREATE SEQUENCES
# ---------------------------------------------------

input_sequences = []

for line in text.split("\n"):

    tokens = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(tokens)):
        seq = tokens[:i + 1]
        input_sequences.append(seq)

print("\nGenerated Sequences:")
for seq in input_sequences:
    print(seq)

# ---------------------------------------------------
# PADDING
# ---------------------------------------------------

max_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_len,
    padding='pre'
)

print("\nPadded Sequences:")
print(input_sequences)

# ---------------------------------------------------
# SPLIT INTO X AND y
# ---------------------------------------------------

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("\nInput Shape:", X.shape)
print("Output Shape:", y.shape)

# ---------------------------------------------------
# BUILD SIMPLE RNN MODEL
# ---------------------------------------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=total_words,
        output_dim=10,
        input_length=max_len - 1
    )
)

model.add(SimpleRNN(32))

model.add(Dense(total_words, activation='softmax'))

# ---------------------------------------------------
# COMPILE MODEL
# ---------------------------------------------------

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

print("\nTraining Started...\n")

history = model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

print("Training Completed")

# ---------------------------------------------------
# FINAL ACCURACY
# ---------------------------------------------------

final_accuracy = history.history['accuracy'][-1]

print("\nFinal Training Accuracy:",
      round(final_accuracy * 100, 2), "%")

# ---------------------------------------------------
# NEXT WORD PREDICTION FUNCTION
# ---------------------------------------------------

def predict_next_word(seed_text):

    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    predicted = model.predict(
        token_list,
        verbose=0
    )

    predicted_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

# ---------------------------------------------------
# TEST PREDICTIONS
# ---------------------------------------------------

print("\n-----------------------------")
print("NEXT WORD PREDICTIONS")
print("-----------------------------")

test_sentences = [
    "deep learning",
    "machine learning",
    "artificial intelligence",
    "deep neural"
]

for sentence in test_sentences:

    predicted_word = predict_next_word(sentence)

    print(
        "Input:",
        sentence,
        " ---> Predicted Word:",
        predicted_word
    )

# ---------------------------------------------------
# SHORT VS LONG SEQUENCE PREDICTION
# ---------------------------------------------------

print("\n-----------------------------")
print("SHORT vs LONG SEQUENCE")
print("-----------------------------")

short_prediction = predict_next_word(
    "machine learning"
)

long_prediction = predict_next_word(
    "artificial intelligence changes"
)

print("Short Sequence Prediction:")
print(
    "Input: machine learning",
    " ---> Prediction:",
    short_prediction
)

print("\nLong Sequence Prediction:")
print(
    "Input: artificial intelligence changes",
    " ---> Prediction:",
    long_prediction
)
