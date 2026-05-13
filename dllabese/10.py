# GAN using MNIST Dataset
# Easy Python Code

# Objectives:
# 1. Generate synthetic handwritten digits
# 2. Observe improvement during training
# 3. Compare Generator vs Discriminator
# 4. Evaluate generated image quality

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------
# LOAD MNIST DATASET
# ---------------------------------------------------

(X_train, _), (_, _) = mnist.load_data()

# Use subset for faster training
X_train = X_train[:5000]

# Normalize images
X_train = X_train / 127.5 - 1.0

# Reshape images
X_train = X_train.reshape(
    X_train.shape[0],
    28,
    28,
    1
)

print("Training Data Shape:", X_train.shape)

# ---------------------------------------------------
# GENERATOR MODEL
# ---------------------------------------------------

generator = Sequential()

generator.add(Dense(128, input_dim=100))
generator.add(LeakyReLU(0.2))

generator.add(Dense(256))
generator.add(LeakyReLU(0.2))

generator.add(Dense(784, activation='tanh'))

generator.add(Reshape((28, 28, 1)))

print("\nGENERATOR SUMMARY")
generator.summary()

# ---------------------------------------------------
# DISCRIMINATOR MODEL
# ---------------------------------------------------

discriminator = Sequential()

discriminator.add(Flatten(input_shape=(28, 28, 1)))

discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Dense(128))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002),
    metrics=['accuracy']
)

print("\nDISCRIMINATOR SUMMARY")
discriminator.summary()

# ---------------------------------------------------
# COMBINED GAN MODEL
# ---------------------------------------------------

discriminator.trainable = False

gan = Sequential([
    generator,
    discriminator
])

gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002)
)

# ---------------------------------------------------
# TRAINING PARAMETERS
# ---------------------------------------------------

epochs = 3000
batch_size = 64

# ---------------------------------------------------
# TRAIN GAN
# ---------------------------------------------------

print("\n-----------------------------")
print("TRAINING GAN")
print("-----------------------------")

for epoch in range(epochs):

    # ------------------------------------------------
    # TRAIN DISCRIMINATOR
    # ------------------------------------------------

    # Select real images
    idx = np.random.randint(
        0,
        X_train.shape[0],
        batch_size
    )

    real_images = X_train[idx]

    # Generate fake images
    noise = np.random.normal(
        0,
        1,
        (batch_size, 100)
    )

    fake_images = generator.predict(
        noise,
        verbose=0
    )

    # Labels
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(
        real_images,
        real_labels
    )

    d_loss_fake = discriminator.train_on_batch(
        fake_images,
        fake_labels
    )

    d_loss = 0.5 * np.add(
        d_loss_real,
        d_loss_fake
    )

    # ------------------------------------------------
    # TRAIN GENERATOR
    # ------------------------------------------------

    noise = np.random.normal(
        0,
        1,
        (batch_size, 100)
    )

    valid_labels = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(
        noise,
        valid_labels
    )

    # ------------------------------------------------
    # PRINT PROGRESS
    # ------------------------------------------------

    if epoch % 500 == 0:

        print("\nEpoch:", epoch)

        print(
            "Discriminator Loss:",
            round(d_loss[0], 4)
        )

        print(
            "Discriminator Accuracy:",
            round(d_loss[1] * 100, 2),
            "%"
        )

        print(
            "Generator Loss:",
            round(float(g_loss), 4)
        )

# ---------------------------------------------------
# GENERATE FINAL IMAGES
# ---------------------------------------------------

print("\n-----------------------------")
print("GENERATING SYNTHETIC IMAGES")
print("-----------------------------")

noise = np.random.normal(
    0,
    1,
    (10, 100)
)

generated_images = generator.predict(
    noise,
    verbose=0
)

# Rescale images
generated_images = 0.5 * generated_images + 0.5

# ---------------------------------------------------
# DISPLAY GENERATED IMAGES
# ---------------------------------------------------

plt.figure(figsize=(10, 4))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        generated_images[i, :, :, 0],
        cmap='gray'
    )

    plt.axis('off')

plt.suptitle("Generated Synthetic Digits")

plt.show()







