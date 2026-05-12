import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:5000]
y_train = y_train[:5000]

x_test = x_test[:1000]
y_test = y_test[:1000]

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Class names
classes = ['Airplane', 'Automobile', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse',
           'Ship', 'Truck']

# -----------------------------
# CNN MODEL
# -----------------------------
cnn_model = Sequential([

    Conv2D(32, (3,3), activation='relu',
           input_shape=(32,32,3)),

    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN Model...\n")

cnn_history = cnn_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test)

print(f"\nCNN Accuracy: {cnn_acc:.4f}")

# -----------------------------
# FULLY CONNECTED MODEL
# -----------------------------
fc_model = Sequential([

    Flatten(input_shape=(32,32,3)),

    Dense(512, activation='relu'),

    Dense(256, activation='relu'),

    Dense(10, activation='softmax')
])

fc_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Fully Connected Model...\n")

fc_history = fc_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

fc_loss, fc_acc = fc_model.evaluate(x_test, y_test)

print(f"\nFully Connected Accuracy: {fc_acc:.4f}")

# -----------------------------
# ACCURACY COMPARISON
# -----------------------------
print("\nFINAL COMPARISON")
print("----------------------")
print(f"CNN Accuracy: {cnn_acc:.4f}")
print(f"FC Accuracy : {fc_acc:.4f}")