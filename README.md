# 🌿 AgriVision - Plant Disease Detection using AI

AgriVision is an AI-powered web application that detects plant diseases from leaf images using deep learning. It works offline and provides accurate predictions along with treatment suggestions.

---

## 🚀 Features

* 🌱 Detects plant diseases from leaf images
* 📷 Capture image using webcam
* 🖼 Upload image from device
* ⚡ Real-time prediction using trained CNN model
* 📊 Shows confidence percentage
* 💡 Provides treatment/solution
* ❌ Rejects non-leaf images (smart validation)
* 🌐 Works completely offline

---

## 🧠 Technologies Used

* Python
* Flask
* TensorFlow / Keras
* OpenCV
* HTML, CSS, JavaScript

---

## 📂 Project Structure

```
AgriVision/
│── app.py
│── predict.py
│── model.h5
│── labels.json
│── train.py
│── dataset/
│── static/
│   ├── style.css
│   ├── script.js
│   └── uploads/
│── templates/
│   └── index.html
│── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/manu-2006/agrivision.git
cd agrivision
```

---

### 2. Create Virtual Environment

```
python -m venv train_env
train_env\Scripts\activate   
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Run the Application

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 📸 How It Works

1. Upload or capture a leaf image
2. System validates if it is a leaf
3. AI model predicts disease
4. Displays:

   * Disease name
   * Confidence %
   * Suggested solution

---

## 🧪 Model Details

* CNN-based deep learning model
* Trained on plant disease dataset
* Input size: 224x224
* Output: Disease classification

---

## ❗ Error Handling

* Detects non-leaf images
* Rejects low confidence predictions
* Provides user-friendly feedback

---

## 📈 Future Enhancements

* Mobile App (APK)
* Real-time video detection
* Multi-language support
* Cloud deployment

---

## 👨‍💻 Author

Manu 

---

## ⭐ GitHub

If you like this project, give it a ⭐ on GitHub!
