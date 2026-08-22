// script.js
// ---------
// Handles drawing on the canvas (mouse + touch), the Clear button,
// and sending the drawn image to the Flask backend for prediction.

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");

let isDrawing = false;

// -----------------------------------------------------------------
// Set up the canvas to look like an MNIST image: black background,
// white stroke.
// -----------------------------------------------------------------
function initCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
initCanvas();

// -----------------------------------------------------------------
// Get the correct (x, y) position for both mouse and touch events.
// -----------------------------------------------------------------
function getPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: clientX - rect.left,
    y: clientY - rect.top,
  };
}

function startDrawing(event) {
  isDrawing = true;
  const pos = getPosition(event);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  event.preventDefault();
}

function draw(event) {
  if (!isDrawing) return;
  const pos = getPosition(event);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  event.preventDefault();
}

function stopDrawing() {
  isDrawing = false;
}

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

// Touch events (mobile support)
canvas.addEventListener("touchstart", startDrawing);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDrawing);

// -----------------------------------------------------------------
// Clear button: reset the canvas back to a blank black square.
// -----------------------------------------------------------------
clearBtn.addEventListener("click", () => {
  initCanvas();
  predictionEl.textContent = "-";
  confidenceEl.textContent = "-";
});

// -----------------------------------------------------------------
// Predict button: send the canvas image to the Flask /predict route.
// -----------------------------------------------------------------
predictBtn.addEventListener("click", async () => {
  const imageDataURL = canvas.toDataURL("image/png");

  predictionEl.textContent = "...";
  confidenceEl.textContent = "...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataURL }),
    });

    const result = await response.json();

    if (result.error) {
      predictionEl.textContent = "Error";
      confidenceEl.textContent = result.error;
      return;
    }

    predictionEl.textContent = result.digit;
    confidenceEl.textContent = `${result.confidence}%`;
  } catch (error) {
    predictionEl.textContent = "Error";
    confidenceEl.textContent = "Could not reach server";
    console.error(error);
  }
});
