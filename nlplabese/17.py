import nltk
import numpy as np
from nltk.corpus import movie_reviews
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Download dataset
nltk.download('movie_reviews')

# Load reviews and labels
reviews = []
labels = []

for fileid in movie_reviews.fileids():
    reviews.append(movie_reviews.raw(fileid))
    labels.append(1 if movie_reviews.categories(fileid)[0] == 'pos' else 0)

# Convert text to numbers
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)

X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X, maxlen=200)

y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build RNN model
model = Sequential()
model.add(Embedding(5000, 32, input_length=200))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)