import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import os
import gdown

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("😊 Emotion Detection App")

classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "emotion_model.pth"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():

    file_id = "1C8l-OBBP_TY4UnTHmwLdUAGkehQBgDBx"
    url = "https://huggingface.co/hiral20/emotion-model/resolve/main/emotion_model.pth"

    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    # Check file size (avoid corrupted download)
    if os.path.getsize(MODEL_PATH) < 10000000:
        st.error("❌ Model download failed. Check Drive sharing.")
        st.stop()

    # Load model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


model = load_model()

# -----------------------------
# TRANSFORM (LIGHTWEIGHT)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(face, model):

    face = cv2.resize(face, (64, 64))

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
            face = img[y:y+h, x:x+w]

            pred, conf = predict(face, model)
            emotion = classes[pred]

            # Draw bounding box
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.putText(img,
                        f"{emotion} ({conf*100:.1f}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

        st.image(img, channels="BGR")
