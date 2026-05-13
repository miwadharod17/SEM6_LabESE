# GRU vs LSTM
# Fast Sequence Model Comparison
# Dataset Used: Small Text Dataset (~1000 chars)

# Objectives:
# 1. Develop faster sequence model (GRU)
# 2. Compare training time with LSTM
# 3. Analyse efficiency vs accuracy
# 4. Evaluate real-time suitability

import numpy as np
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# TEXT DATASET (~1000 chars)
# ---------------------------------------------------

text = """
artificial intelligence is transforming industries
machine learning helps computers learn patterns
deep learning models process large amounts of data
natural language processing understands human language
chatbots communicate with users automatically
recurrent neural networks process sequential information
gru models train faster than lstm models
lstm networks remember long term dependencies
sequence prediction is important in deep learning
python is widely used for artificial intelligence
tensorflow simplifies neural network implementation
keras provides high level deep learning tools
students learn machine learning for modern applications
computer vision detects objects in images
speech recognition converts audio into text
recommendation systems improve user experience
cybersecurity systems identify network attacks
data science combines statistics and programming
optimization algorithms reduce model loss
real time applications require faster prediction
"""

# ---------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# ---------------------------------------------------
# CREATE INPUT SEQUENCES
# ---------------------------------------------------

input_sequences = []

for line in text.split("\n"):

    tokens = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(tokens)):

        seq = tokens[:i + 1]
        input_sequences.append(seq)

# ---------------------------------------------------
# PAD SEQUENCES
# ---------------------------------------------------

max_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_len,
    padding='pre'
)

# ---------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# ---------------------------------------------------
# BUILD GRU MODEL
# ---------------------------------------------------

gru_model = Sequential()

gru_model.add(
    Embedding(
        input_dim=total_words,
        output_dim=10,
        input_length=max_len - 1
    )
)

gru_model.add(GRU(32))

gru_model.add(
    Dense(
        total_words,
        activation='softmax'
    )
)

gru_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------------------------------------------------
# TRAIN GRU MODEL
# ---------------------------------------------------

print("\n-----------------------------")
print("TRAINING GRU MODEL")
print("-----------------------------")

start_time = time.time()

gru_history = gru_model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

gru_training_time = time.time() - start_time

# ---------------------------------------------------
# BUILD LSTM MODEL
# ---------------------------------------------------

lstm_model = Sequential()

lstm_model.add(
    Embedding(
        input_dim=total_words,
        output_dim=10,
        input_length=max_len - 1
    )
)

lstm_model.add(LSTM(32))

lstm_model.add(
    Dense(
        total_words,
        activation='softmax'
    )
)

lstm_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------------------------------------------------
# TRAIN LSTM MODEL
# ---------------------------------------------------

print("\n-----------------------------")
print("TRAINING LSTM MODEL")
print("-----------------------------")

start_time = time.time()

lstm_history = lstm_model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

lstm_training_time = time.time() - start_time

# ---------------------------------------------------
# ACCURACY COMPARISON
# ---------------------------------------------------

gru_acc = gru_history.history['accuracy'][-1]
lstm_acc = lstm_history.history['accuracy'][-1]

print("\n-----------------------------")
print("ACCURACY COMPARISON")
print("-----------------------------")

print("GRU Accuracy:",
      round(gru_acc * 100, 2), "%")

print("LSTM Accuracy:",
      round(lstm_acc * 100, 2), "%")

# ---------------------------------------------------
# LOSS COMPARISON
# ---------------------------------------------------

gru_loss = gru_history.history['loss'][-1]
lstm_loss = lstm_history.history['loss'][-1]

print("\n-----------------------------")
print("LOSS COMPARISON")
print("-----------------------------")

print("GRU Loss:",
      round(gru_loss, 4))

print("LSTM Loss:",
      round(lstm_loss, 4))

# ---------------------------------------------------
# TRAINING TIME COMPARISON
# ---------------------------------------------------

print("\n-----------------------------")
print("TRAINING TIME COMPARISON")
print("-----------------------------")

print("GRU Training Time:",
      round(gru_training_time, 2),
      "seconds")

print("LSTM Training Time:",
      round(lstm_training_time, 2),
      "seconds")

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_next_word(model, seed_text):

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
# PREDICTION TEST
# ---------------------------------------------------

print("\n-----------------------------")
print("PREDICTION TEST")
print("-----------------------------")

test_text = "artificial intelligence"

gru_prediction = predict_next_word(
    gru_model,
    test_text
)

lstm_prediction = predict_next_word(
    lstm_model,
    test_text
)

print("Input:", test_text)

print("\nGRU Prediction:",
      gru_prediction)

print("LSTM Prediction:",
      lstm_prediction)


