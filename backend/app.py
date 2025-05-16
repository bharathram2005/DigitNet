from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Dummy CNN (replace with your own)
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# Load model (replace with your trained model)
model = DummyCNN()
model.eval()

# Transform input to match MNIST expectations
transform = transforms.Compose([
    transforms.Grayscale(),       # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to MNIST shape
    transforms.ToTensor(),        # Convert to tensor [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to match training
])

@app.route("/")
def index():
    return "✅ Flask backend is running."


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)  # shape: [1, 1, 28, 28]
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        top3 = torch.topk(probs, 3)

    response = {
        "predictions": [
            {"digit": int(top3.indices[0][i]), "confidence": float(top3.values[0][i])}
            for i in range(3)
        ]
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
