import numpy as np
import json
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Load model
model = load_model("model.h5")

# ✅ Load labels
with open("labels.json", "r") as f:
    class_names = json.load(f)

# ✅ Disease solutions
solutions = {
    "Tomato_Early_blight": "Use fungicide and remove infected leaves",
    "Tomato_Late_blight": "Apply copper-based fungicide",
    "Tomato_healthy": "Plant is healthy, no action needed",
    "Potato_Early_blight": "Use fungicide spray",
    "Potato_Late_blight": "Avoid excess moisture and use fungicide",
    "Corn_Common_rust": "Use resistant varieties and fungicide",
    "Corn_healthy": "Healthy plant"
}

# 🔥 LEAF DETECTION FUNCTION
def is_leaf_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green color range (leaf detection)
    lower_green = (25, 40, 40)
    upper_green = (90, 255, 255)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]

    green_ratio = green_pixels / total_pixels

    # 🔥 threshold (adjust if needed)
    return green_ratio > 0.2


# 🔥 MAIN PREDICTION FUNCTION
def predict_disease(img_path):

    # ❗ STEP 1: Check if image is leaf
    if not is_leaf_image(img_path):
        return "Invalid Image", 0, "Please upload a proper leaf image"

    # ✅ STEP 2: Load image for model
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # ✅ STEP 3: Predict
    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    class_index = int(np.argmax(prediction))

    label = class_names[str(class_index)]

    # ❗ STEP 4: Confidence validation
    if confidence < 0.85:
        return "Invalid Image", confidence, "Please upload a proper leaf image"

    # ✅ STEP 5: Get solution
    solution = solutions.get(label, "Consult agricultural expert")

    return label, confidence, solution