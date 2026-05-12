import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATASET
# -----------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Use small subset for faster training
x_train = x_train[:5000]
y_train = y_train[:5000]

x_test = x_test[:1000]
y_test = y_test[:1000]

# Resize images for VGG16
x_train = tf.image.resize(x_train, (64,64))
x_test = tf.image.resize(x_test, (64,64))

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# -----------------------------
# LOAD PRETRAINED VGG16
# -----------------------------

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(64,64,3)
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# CREATE MODEL
# -----------------------------

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------

loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")

