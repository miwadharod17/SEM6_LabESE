import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATASET
# -----------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Small subset for faster training
x_train = x_train[:5000]
y_train = y_train[:5000]

x_test = x_test[:1000]
y_test = y_test[:1000]

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# -----------------------------
# FUNCTION TO CREATE CNN
# -----------------------------

def create_model(learning_rate):

    model = Sequential([

        Conv2D(32, (3,3),
               activation='relu',
               input_shape=(32,32,3)),

        MaxPooling2D((2,2)),

        Conv2D(64, (3,3),
               activation='relu'),

        MaxPooling2D((2,2)),

        Flatten(),

        Dense(64, activation='relu'),

        Dense(10, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# -----------------------------
# DIFFERENT LEARNING RATES
# -----------------------------

learning_rates = [0.1, 0.001]

results = []

for lr in learning_rates:

    print("\n---------------------------")
    print(f"Learning Rate: {lr}")

    model = create_model(lr)

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test)
    )

    loss, acc = model.evaluate(x_test, y_test)

    results.append((lr, acc))

# -----------------------------
# FINAL RESULTS
# -----------------------------

print("\nFINAL RESULTS")
print("---------------------------")

for r in results:
    print(f"Learning Rate: {r[0]} | Accuracy: {r[1]:.4f}")