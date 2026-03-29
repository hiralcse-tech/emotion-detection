import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import requests
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("😊 Emotion Detection (Improved Accuracy)")

classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "emotion_model.pth"

# -----------------------------
# LOAD MODEL (HUGGINGFACE)
# -----------------------------
@st.cache_resource
def load_model():

    url = "https://huggingface.co/hiral20/emotion-model/resolve/main/emotion_model.pth"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(url)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    if os.path.getsize(MODEL_PATH) < 10000000:
        st.error("❌ Model download failed")
        st.stop()

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


model = load_model()

# -----------------------------
# BETTER TRANSFORM (IMPORTANT)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # improved from 64
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],   # ImageNet mean
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
# PREPROCESS FACE (ENHANCED)
# -----------------------------
def preprocess_face(face):

    # Resize
    face = cv2.resize(face, (96, 96))

    # Sharpen image (improves features)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    face = cv2.filter2D(face, -1, kernel)

    return face

# -----------------------------
# PREDICT
# -----------------------------
def predict(face, model):

    face = preprocess_face(face)

    img = Image.fromarray(face)
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()

# -----------------------------
# UI
# -----------------------------
st.subheader("📷 Upload Image")

file = st.file_uploader("Upload an image", type=["jpg", "png"])

if file is not None:

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected 😢")
        st.image(img, channels="BGR")
    else:
        for (x, y, w, h) in faces:

            # Padding (IMPORTANT for better context)
            pad = 15
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            face = img[y1:y2, x1:x2]

            pred, conf = predict(face, model)

            # Confidence filter
            if conf < 0.4:
                emotion = "Uncertain"
            else:
                emotion = classes[pred]

            # Draw box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.putText(img,
                        f"{emotion} ({conf*100:.1f}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

        st.image(img, channels="BGR")
