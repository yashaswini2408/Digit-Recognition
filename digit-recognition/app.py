"""
app.py
------
Flask backend for the Handwritten Digit Recognition web app.

It loads the trained CNN model and exposes a /predict endpoint that
accepts a canvas drawing (as a base64 PNG image), preprocesses it into
MNIST format, and returns the predicted digit and confidence score.

Run with:
    python app.py
"""

import base64
import io
import re

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# -----------------------------------------------------------------
# Load the trained model once, when the server starts.
# -----------------------------------------------------------------
print("Loading trained model...")
model = tf.keras.models.load_model("mnist_model.keras")
print("Model loaded successfully.")


def preprocess_image(image_data_url):
    """
    Convert the base64 image data URL coming from the canvas into a
    28x28 normalized numpy array, matching the MNIST format.
    """
    # The data URL looks like "data:image/png;base64,iVBORw0KG..."
    # Strip the header so we're left with just the base64 data.
    image_data = re.sub("^data:image/.+;base64,", "", image_data_url)
    image_bytes = base64.b64decode(image_data)

    # Open the image and convert it to grayscale ("L" mode).
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Resize to 28x28 pixels, the size MNIST images use.
    image = image.resize((28, 28))

    # Convert to a numpy array and normalize pixel values to 0-1.
    image_array = np.array(image).astype("float32") / 255.0

    # Reshape into the format the model expects: (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a drawn digit image from the frontend, run it through the
    trained model, and return the predicted digit + confidence.
    """
    data = request.get_json()
    image_data_url = data.get("image")

    if not image_data_url:
        return jsonify({"error": "No image data received"}), 400

    # Preprocess the incoming canvas image to match MNIST format.
    processed_image = preprocess_image(image_data_url)

    # Run the prediction.
    predictions = model.predict(processed_image)
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100

    return jsonify({
        "digit": predicted_digit,
        "confidence": round(confidence, 2),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
