from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import base64
import io
from predict import predict_disease

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.json") as f:
    class_names = json.load(f)

# Preprocess (MUST MATCH TRAINING)
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Solutions dictionary
solutions = {
    "Potato___Early_blight": "Apply fungicide and remove infected leaves",
    "Potato___Late_blight": "Use resistant seeds and avoid excess moisture",
    "Tomato___Bacterial_spot": "Use copper sprays and remove infected leaves",
    "Tomato___Early_blight": "Apply fungicide and improve air circulation",
    "Tomato___Late_blight": "Avoid overhead watering and use resistant varieties",
    "Tomato___Leaf_Mold": "Reduce humidity and improve ventilation",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicide",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Use insecticidal soap",
    "Tomato___Target_Spot": "Apply fungicide",
    "Tomato___Tomato_YellowLeaf_Curl_Virus": "Control whiteflies",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants",
    "Tomato___healthy": "Plant is healthy 🌱",
    "Potato___healthy": "Plant is healthy 🌱",
    "Pepper__bell___healthy": "Plant is healthy 🌱"
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ✅ call your ML function
    label, confidence, solution = predict_disease(filepath)

    return render_template(
        "index.html",
        prediction=label,
        confidence=round(confidence * 100, 2),
        solution=solution,
        image_path=filepath,
        is_invalid=(label == "Invalid Image")
    )

@app.route('/camera_predict', methods=['POST'])
def camera_predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]

    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    filepath = "static/captured.png"
    image.save(filepath)

    label, confidence = predict_disease(filepath)

    return {
        "label": label,
        "confidence": str(round(confidence * 100, 2)) + "%"
    }
if __name__ == "__main__":
    app.run(debug=True)