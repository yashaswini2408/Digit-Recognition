"""
train_model.py
----------------
This script trains a simple CNN (Convolutional Neural Network) on the
MNIST handwritten digit dataset and saves the trained model to disk.

Run this once before starting the Flask app:
    python train_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------------------------------------------
# 1. Load the MNIST dataset
# -----------------------------------------------------------------
# MNIST is built into Keras, so it downloads automatically the
# first time you run this script (needs an internet connection).
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# -----------------------------------------------------------------
# 2. Preprocess the data
# -----------------------------------------------------------------
# Normalize pixel values from the 0-255 range to 0-1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# The CNN expects images shaped as (28, 28, 1) -> add a channel dimension.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# -----------------------------------------------------------------
# 3. Build a simple CNN model
# -----------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),  # 10 classes: digits 0-9
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------------------------------------------
# 4. Train the model
# -----------------------------------------------------------------
print("\nTraining model...")
model.fit(
    x_train, y_train,
    epochs=5,                 # 5 epochs is enough for good accuracy on MNIST
    batch_size=128,
    validation_split=0.1,
)

# -----------------------------------------------------------------
# 5. Evaluate on the test set
# -----------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

# -----------------------------------------------------------------
# 6. Save the trained model
# -----------------------------------------------------------------
model.save("mnist_model.keras")
print("\nModel saved as mnist_model.keras")
