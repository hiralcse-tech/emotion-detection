import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import os
import requests
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Emotion AI + Grad-CAM", layout="centered")
st.title("😊 Emotion Detection + Explainable AI")

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

    # Load state_dict
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model

model = load_model()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------------
# FACE DETECTOR + ALIGNMENT
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh

def align_face(image, box):
    x, y, w, h = box
    face = image[y:y+h, x:x+w]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_eye = lm[33]
            right_eye = lm[263]
            dy = right_eye.y - left_eye.y
            dx = right_eye.x - left_eye.x
            angle = np.arctan2(dy, dx) * 180 / np.pi
            center = (x + w//2, y + h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            face = aligned[y:y+h, x:x+w]
    return face

# -----------------------------
# PREDICT + GRADCAM
# -----------------------------
def predict(face):
    face_rgb = cv2.cvtColor(cv2.resize(face,(224,224)), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face_rgb)
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs,1)

    return pred.item(), conf.item(), img_t

def generate_gradcam(model, image_tensor, target_class, face_size):
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
    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i,w in enumerate(weights):
        cam += w*acts[i]
    cam = np.maximum(cam,0)
    cam = cv2.resize(cam, face_size)
    if cam.max() !=0:
        cam = cam/cam.max()
    return cam

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.subheader("📷 Upload Images")
files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

if files:
    for file in files:
        st.markdown("---")
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces)==0:
            st.warning("No face detected 😢")
            st.image(img, channels="BGR")
            continue

        for i,(x,y,w,h) in enumerate(faces):
            face = align_face(img,(x,y,w,h))
            pred, conf, img_t = predict(face)
            emotion = classes[pred]

            # Grad-CAM
            cam = generate_gradcam(model, img_t, pred, (w,h))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(face, 0.6, heatmap,0.4,0)

            # Draw box + label
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,f"{emotion} ({conf*100:.1f}%)",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            st.image(overlay, caption=f"🔥 Grad-CAM Face {i+1}", channels="BGR")
        st.image(img, channels="BGR")
