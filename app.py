import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import gdown
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Emotion AI", layout="centered")
st.title("😊 Emotion Detection + Explainable AI")

classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "emotion_model.pth"

# -----------------------------
# LOAD MODEL (FIXED)
# -----------------------------
@st.cache_resource
def load_model():

    # Re-download if file missing or corrupted
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        url = "https://drive.google.com/uc?id=1wYbI3OxE0yktvwArreqN0PedlALKE8ru"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model

model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((96, 96)),
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
# GRAD-CAM
# -----------------------------
def generate_gradcam(model, image_tensor, target_class):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = model.layer4[-1].conv2
    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    loss = output[0, target_class]

    model.zero_grad()
    loss.backward()

    grads = gradients[0].cpu().detach().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (96, 96))

    if cam.max() != 0:
        cam = cam / cam.max()

    return cam

# -----------------------------
# PREDICTION
# -----------------------------
def predict(face):

    face = cv2.resize(face, (96, 96))

    img = Image.fromarray(face)
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item(), img_t

# -----------------------------
# UI - IMAGE UPLOAD
# -----------------------------
st.subheader("📷 Upload Image")

files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

if files:
    for file in files:
        st.markdown("---")

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("No face detected 😢")
            st.image(img, channels="BGR")
            continue

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]

            pred, conf, img_t = predict(face)
            emotion = classes[pred]

            # Grad-CAM
            cam = generate_gradcam(model, img_t, pred)
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (w,h))

            overlay = cv2.addWeighted(face, 0.6, heatmap, 0.4, 0)

            # Draw bounding box
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, f"{emotion} ({conf*100:.1f}%)",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

            st.image(overlay, caption="🔥 Grad-CAM", channels="BGR")

        st.image(img, channels="BGR")
