import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import time

# -----------------------------
# LOAD DATASET
# -----------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Use small subset
x_train = x_train[:5000]
y_train = y_train[:5000]

x_test = x_test[:1000]
y_test = y_test[:1000]

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# =============================
# CNN FROM SCRATCH
# =============================

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

print("\nTraining CNN From Scratch...\n")

start = time.time()

cnn_history = cnn_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

cnn_time = time.time() - start

cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test)

# =============================
# TRANSFER LEARNING (VGG16)
# =============================

# Resize images for VGG16
x_train_vgg = tf.image.resize(x_train, (64,64))
x_test_vgg = tf.image.resize(x_test, (64,64))

# Load pretrained VGG16
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(64,64,3)
)

# Freeze layers
base_model.trainable = False

transfer_model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

transfer_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Transfer Learning Model...\n")

start = time.time()

transfer_history = transfer_model.fit(
    x_train_vgg,
    y_train,
    epochs=5,
    validation_data=(x_test_vgg, y_test)
)

transfer_time = time.time() - start

transfer_loss, transfer_acc = transfer_model.evaluate(
    x_test_vgg,
    y_test
)

# =============================
# FINAL RESULTS
# =============================

print("\nFINAL COMPARISON")
print("--------------------------------")

print(f"CNN Accuracy              : {cnn_acc:.4f}")
print(f"Transfer Learning Accuracy: {transfer_acc:.4f}")

print(f"\nCNN Training Time         : {cnn_time:.2f} sec")
print(f"Transfer Learning Time    : {transfer_time:.2f} sec")
