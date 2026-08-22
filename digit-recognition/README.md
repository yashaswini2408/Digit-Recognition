# Handwritten Digit Recognition Web App

A simple web app where you draw a digit (0–9) on a canvas and a trained
deep-learning model (CNN) predicts what you drew.

## 1. Project Abstract

This mini-project demonstrates a full end-to-end deep learning pipeline:
training a Convolutional Neural Network (CNN) on the MNIST dataset,
then serving that model through a Flask web app so a user can draw a
digit in the browser and get a live prediction with a confidence score.

## 2. Objectives

- Train a CNN to classify handwritten digits (0–9) using MNIST.
- Build a simple web interface with a drawing canvas.
- Connect the frontend to a Flask backend that runs the model.
- Return the predicted digit and confidence score to the user.

## 3. Technologies Used

- **Python** – core language
- **TensorFlow/Keras** – building and training the CNN
- **MNIST dataset** – 60,000 training + 10,000 test handwritten digit images
- **Flask** – backend web server and API
- **HTML/CSS/JavaScript** – frontend UI and canvas drawing
- **Pillow (PIL)** – image processing on the backend
- **NumPy** – array handling

## 4. Project Structure

```
digit-recognition/
│
├── app.py              # Flask backend, serves the page and /predict API
├── train_model.py       # Script to train and save the CNN model
├── requirements.txt      # Python dependencies
│
├── templates/
│   └── index.html       # Main web page
│
└── static/
    ├── style.css         # Page styling
    └── script.js         # Canvas drawing + calls to /predict
```

Note: `mnist_model.keras` is NOT included in this zip — you generate it
yourself in step 2 below by running the training script. This keeps the
zip small and lets the model train fresh on your machine.

## 5. Installation

1. Make sure you have **Python 3.9+** installed.
2. Unzip this project and open a terminal inside the `digit-recognition`
   folder.
3. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 6. Train the Model

Run this once. It downloads MNIST automatically (needs internet the
first time) and saves the trained model as `mnist_model.keras`.

```bash
python train_model.py
```

This takes a few minutes on a normal laptop CPU. At the end you'll see
something like:

```
Test accuracy: 99.1%
Model saved as mnist_model.keras
```

## 7. Run the Web App

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Draw a digit on the black canvas, click **Predict**, and see the
result. Click **Clear** to try again.

## 8. How It Works (Simple Explanation)

1. We train a small neural network (a CNN) on 60,000 example images of
   handwritten digits so it learns what each digit "looks like".
2. When you draw a digit on the webpage, JavaScript captures your
   drawing as an image.
3. That image is sent to the Flask backend, which resizes it to
   28×28 pixels and normalizes it — the same format the model was
   trained on.
4. The trained model looks at the image and outputs a probability for
   each digit (0–9). The highest probability is the predicted digit,
   and that probability (as a percentage) is the confidence score.

## 9. Possible Viva Questions & Answers

**Q: What is MNIST?**
A: A dataset of 70,000 grayscale images (28×28 pixels) of handwritten
digits 0–9, commonly used to learn/benchmark image classification.

**Q: Why normalize pixel values to 0–1?**
A: Neural networks train faster and more reliably when input values
are on a small, consistent scale rather than 0–255.

**Q: What is a CNN and why use it here?**
A: A Convolutional Neural Network is a type of neural network that
uses filters to detect visual patterns like edges and curves, making
it very effective for image data like handwritten digits.

**Q: What do Conv2D and MaxPooling2D layers do?**
A: Conv2D layers scan the image with small filters to detect features
(edges, curves). MaxPooling2D shrinks the feature maps, keeping the
strongest signals and reducing computation.

**Q: What does the Dropout layer do?**
A: It randomly turns off some neurons during training to prevent the
model from memorizing the training data (overfitting).

**Q: Why is the last layer's activation "softmax"?**
A: Softmax turns the model's raw outputs into probabilities across the
10 digit classes that sum to 1, so we can pick the most likely digit.

**Q: What does the /predict endpoint do?**
A: It receives the canvas image as base64, converts it to grayscale,
resizes it to 28x28, normalizes it, feeds it to the model, and returns
the predicted digit and confidence as JSON.

**Q: Why resize the canvas drawing to 28×28?**
A: Because the model was trained on 28×28 MNIST images, so the input
must match that exact size and format.

**Q: What is confidence score?**
A: The probability (from softmax) the model assigned to its predicted
digit — how "sure" it is about that answer.

## 10. Ideas for Simple Improvements

- Show a bar chart of probabilities for all 10 digits, not just the top one.
- Let users save/download their drawing.
- Add a "history" of past predictions on the page.
- Try a slightly deeper CNN or more training epochs to improve accuracy.
- Add a dark/light theme toggle.
