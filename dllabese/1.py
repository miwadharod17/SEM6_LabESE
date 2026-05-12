import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize values (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Function to create model
def create_model(hidden_layers=1, activation='relu'):

    model = Sequential()

    # Convert 28x28 image into 784 inputs
    model.add(Flatten(input_shape=(28, 28)))

    # Add hidden layers
    for i in range(hidden_layers):
        model.add(Dense(128, activation=activation))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    return model


# Different activation functions
activations = ['relu', 'sigmoid', 'tanh']

# Different hidden layer counts
layers_list = [1, 2, 3]

results = []

for act in activations:
    for layers in layers_list:

        print("\n-----------------------------------")
        print(f"Activation: {act}")
        print(f"Hidden Layers: {layers}")

        # Create model
        model = create_model(hidden_layers=layers,
                             activation=act)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            x_train,
            y_train,
            epochs=5,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test)

        print(f"Test Accuracy: {test_acc:.4f}")

        # Store results
        results.append((act, layers, test_acc))

# Print final comparison
print("\nFINAL RESULTS")
print("-----------------------------------")

for r in results:
    print(f"Activation: {r[0]} | Layers: {r[1]} | Accuracy: {r[2]:.4f}")