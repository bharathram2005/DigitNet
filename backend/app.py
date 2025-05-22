from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from flask_cors import CORS
from preprocessing import preprocess_image

#import our trained model
from model import DigitCNN

app = Flask(__name__)
CORS(app)


model = DigitCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


# Transform input to match MNIST expectations
transform = transforms.Compose([
    transforms.Grayscale(),       # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to MNIST shape
    transforms.ToTensor(),        # Convert to tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize to match training
])

@app.route("/")
def index():
    return "✅ Flask backend is running."


@app.route("/predict", methods=["POST"])
def predict():
    print("backend /predict route is running now")
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Preprocess and predict
    #img_tensor = transform(image).unsqueeze(0)  # shape: [1, 1, 28, 28]
    img_tensor = preprocess_image(image)
    if img_tensor is None:
        return jsonify({"error": "Empty drawing"}), 400

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
