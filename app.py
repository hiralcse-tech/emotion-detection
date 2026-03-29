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
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("😊 Emotion Detection System (Aligned Faces)")

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

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    st.success("✅ Model loaded successfully")
    model.eval()
    return model

model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# FACE DETECTION + ALIGNMENT
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

def align_face(image, box):
    x, y, w, h = box
    face = image[y:y+h, x:x+w]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            # Use first face landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]  # approx left eye
            right_eye = landmarks[263]  # approx right eye

            # Compute rotation angle
            eye_dx = right_eye.x - left_eye.x
            eye_dy = right_eye.y - left_eye.y
            angle = np.arctan2(eye_dy, eye_dx) * 180.0 / np.pi

            # Rotate the whole image around the face center
            center = (x + w//2, y + h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            face = aligned[y:y+h, x:x+w]

    return face

# -----------------------------
# PREDICT
# -----------------------------
def predict(face, model):
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
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
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected 😢")
        st.image(img, channels="BGR")
    else:
        for i, (x, y, w, h) in enumerate(faces):
            face = align_face(img, (x, y, w, h))

            probs = predict(face, model)
            conf, pred = torch.max(probs, 0)

            emotion = classes[pred] if conf > 0.4 else "Uncertain"

            # Draw box & label
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,
                        f"{emotion} ({conf*100:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            # Display each face’s probability chart
            st.subheader(f"📊 Emotion Probabilities (Face {i+1})")
            df = pd.DataFrame({"Emotion": classes, "Confidence": probs.numpy()})
            st.bar_chart(df.set_index("Emotion"))

            st.subheader(f"🧠 Top 3 Predictions (Face {i+1})")
            top3 = torch.topk(probs, 3)
            for j in range(3):
                st.write(f"{classes[top3.indices[j]]} → {top3.values[j]*100:.2f}%")

        st.image(img, channels="BGR")
