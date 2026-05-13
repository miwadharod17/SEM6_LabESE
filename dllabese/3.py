import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import time

# -----------------------------
# LOAD DATASET
# -----------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Small subset
x_train = x_train[:1000]
y_train = y_train[:1000]

x_test = x_test[:200]
y_test = y_test[:200]

# Resize images
x_train = tf.image.resize(x_train, (32,32))
x_test = tf.image.resize(x_test, (32,32))

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# =====================================================
# MODEL 1 : FEATURE EXTRACTION (FROZEN LAYERS)
# =====================================================

base_model1 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3)
)

# Freeze all layers
base_model1.trainable = False

model1 = Sequential([
    base_model1,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Frozen Layer Model...\n")

start = time.time()

history1 = model1.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

time1 = time.time() - start

loss1, acc1 = model1.evaluate(x_test, y_test)

# =====================================================
# MODEL 2 : PARTIAL FINE TUNING
# =====================================================

base_model2 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3)
)

# Enable training
base_model2.trainable = True

# Freeze first layers only
for layer in base_model2.layers[:15]:
    layer.trainable = False

model2 = Sequential([
    base_model2,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Fine-Tuning Model...\n")

start = time.time()

history2 = model2.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

time2 = time.time() - start

loss2, acc2 = model2.evaluate(x_test, y_test)

# =====================================================
# FINAL COMPARISON
# =====================================================

print("\nFINAL COMPARISON")
print("--------------------------------")

print(f"Frozen Layers Accuracy     : {acc1:.4f}")
print(f"Fine-Tuning Accuracy       : {acc2:.4f}")

print(f"\nFrozen Layers Time         : {time1:.2f} sec")
print(f"Fine-Tuning Time           : {time2:.2f} sec")

# =====================================================
# PLOT GRAPH
# =====================================================

plt.plot(history1.history['accuracy'],
         label='Frozen Layers')

plt.plot(history2.history['accuracy'],
         label='Fine Tuning')

plt.title("Frozen Layers vs Fine Tuning")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()