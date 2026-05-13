# LSTM vs Simple RNN
# Easy Python Code using TensorFlow/Keras

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# TEXT DATA
# ---------------------------------------------------

# Bigger Dataset for LSTM vs Simple RNN
# Copy this directly into your code

text = """
deep learning is transforming technology
deep learning models learn complex patterns
deep learning improves image recognition
deep learning helps in speech processing
machine learning is widely used today
machine learning algorithms analyze data
machine learning improves prediction accuracy
machine learning powers recommendation systems
artificial intelligence is changing industries
artificial intelligence improves automation
artificial intelligence supports decision making
artificial intelligence is used in healthcare
natural language processing understands text
natural language processing enables chatbots
natural language processing helps translation
recurrent neural networks process sequential data
recurrent neural networks remember previous inputs
simple rnn suffers from vanishing gradients
lstm solves long term dependency problems
lstm networks use memory cells
lstm models work well on text prediction
gated recurrent units are similar to lstm
neural networks consist of interconnected neurons
convolutional neural networks process images
transformer models improve language understanding
data science combines statistics and programming
big data analytics extracts useful insights
python is popular for machine learning
tensorflow is used for deep learning
keras simplifies neural network development
sequence models predict future values
time series forecasting uses sequential models
students learn artificial intelligence concepts
deep neural networks require large datasets
optimization algorithms improve model performance
gradient descent minimizes loss functions
computer vision detects objects in images
speech recognition converts audio into text
chatbots interact with users automatically
recommendation systems suggest useful products
self driving cars use artificial intelligence
medical diagnosis systems use machine learning
financial prediction uses deep learning models
cybersecurity systems detect network attacks
robotics uses artificial intelligence techniques
language models generate human like text
text classification categorizes documents automatically
sentiment analysis identifies user opinions
sequence learning is important in deep learning
memory mechanisms improve long sequence prediction
lstm retains important past information
simple rnn forgets old information quickly
"""

# ---------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Total Words:", total_words)

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
# SPLIT INTO X AND y
# ---------------------------------------------------

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("Input Shape:", X.shape)
print("Output Shape:", y.shape)

# ---------------------------------------------------
# SIMPLE RNN MODEL
# ---------------------------------------------------

rnn_model = Sequential()

rnn_model.add(
    Embedding(
        input_dim=total_words,
        output_dim=10,
        input_length=max_len - 1
    )
)

rnn_model.add(SimpleRNN(32))

rnn_model.add(
    Dense(
        total_words,
        activation='softmax'
    )
)

rnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n-----------------------------")
print("TRAINING SIMPLE RNN")
print("-----------------------------")

rnn_history = rnn_model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

# ---------------------------------------------------
# LSTM MODEL
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

print("\n-----------------------------")
print("TRAINING LSTM")
print("-----------------------------")

lstm_history = lstm_model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

# ---------------------------------------------------
# FINAL ACCURACY COMPARISON
# ---------------------------------------------------

rnn_acc = rnn_history.history['accuracy'][-1]
lstm_acc = lstm_history.history['accuracy'][-1]

print("\n-----------------------------")
print("ACCURACY COMPARISON")
print("-----------------------------")

print("Simple RNN Accuracy:",
      round(rnn_acc * 100, 2), "%")

print("LSTM Accuracy:",
      round(lstm_acc * 100, 2), "%")

# ---------------------------------------------------
# LOSS COMPARISON
# ---------------------------------------------------

rnn_loss = rnn_history.history['loss'][-1]
lstm_loss = lstm_history.history['loss'][-1]

print("\n-----------------------------")
print("LOSS COMPARISON")
print("-----------------------------")

print("Simple RNN Loss:",
      round(rnn_loss, 4))

print("LSTM Loss:",
      round(lstm_loss, 4))

# ---------------------------------------------------
# NEXT WORD PREDICTION FUNCTION
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
# PREDICTION COMPARISON
# ---------------------------------------------------

print("\n-----------------------------")
print("PREDICTION COMPARISON")
print("-----------------------------")

test_sentence = "deep learning"

rnn_prediction = predict_next_word(
    rnn_model,
    test_sentence
)

lstm_prediction = predict_next_word(
    lstm_model,
    test_sentence
)

print("Input Sentence:", test_sentence)

print("\nSimple RNN Prediction:",
      rnn_prediction)

print("LSTM Prediction:",
      lstm_prediction)

# ---------------------------------------------------
# LONG SEQUENCE TEST
# ---------------------------------------------------

print("\n-----------------------------")
print("LONG SEQUENCE TEST")
print("-----------------------------")

long_text = "artificial intelligence changes"

rnn_long = predict_next_word(
    rnn_model,
    long_text
)

lstm_long = predict_next_word(
    lstm_model,
    long_text
)

print("Input:", long_text)

print("\nSimple RNN Prediction:",
      rnn_long)

print("LSTM Prediction:",
      lstm_long)

