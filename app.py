import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
import requests

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("😊 Emotion Detection System (Enhanced Accuracy)")

classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "emotion_model.pth"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/hiral20/emotion-model/resolve/main/emotion_model.pth"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(url)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    # Create model architecture
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    # Load state_dict only (fixes _pickle error)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    st.success("✅ Model loaded successfully")

    model.eval()
    return model

# Load the model once
model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # larger for better accuracy
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# PREPROCESS FACE
# -----------------------------
def preprocess_face(face):
    # Resize to 224x224 RGB
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(face, model):
    face = preprocess_face(face)
    img = Image.fromarray(face)
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)[0]

    return probs

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.subheader("📷 Upload Image")
file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected 😢")
        st.image(img, channels="BGR")
    else:
        for (x, y, w, h) in faces:

            # Padding
            pad = 15
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            face = img[y1:y2, x1:x2]

            # Predict emotion
            probs = predict(face, model)
            conf, pred = torch.max(probs, 0)

            # Decide emotion
            if conf < 0.4:
                emotion = "Uncertain"
            else:
                emotion = classes[pred]

            # Draw box & label
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,
                        f"{emotion} ({conf*100:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            # -----------------------------
            # 📊 PROBABILITY CHART
            # -----------------------------
            st.subheader("📊 Emotion Probabilities")
            df = pd.DataFrame({
                "Emotion": classes,
                "Confidence": probs.numpy()
            })
            st.bar_chart(df.set_index("Emotion"))

            # -----------------------------
            # 🧠 TOP 3 PREDICTIONS
            # -----------------------------
            st.subheader("🧠 Top Predictions")
            top3 = torch.topk(probs, 3)
            for i in range(3):
                st.write(
                    f"{classes[top3.indices[i]]} → {top3.values[i]*100:.2f}%"
                )

        # Show annotated image
        st.image(img, channels="BGR")
