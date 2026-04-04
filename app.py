import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import os
import time
import requests

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection with Explainable AI",
    page_icon="😊",
    layout="wide"
)

# -----------------------------
# CONSTANTS - UPDATE THESE TO MATCH YOUR MODEL
# -----------------------------
# Change these values based on how your model was trained
INPUT_SIZE = 48  # Try: 48, 96, or 224
USE_GRAYSCALE = True
NORMALIZE_MEAN = [0.5]  # Usually [0.5] for normalized images
NORMALIZE_STD = [0.5]   # Usually [0.5] for normalized images

# FER2013 Standard Classes (most common)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOJI_MAP = {
    'angry': '😠', 
    'disgust': '🤢', 
    'fear': '😨',
    'happy': '😊', 
    'neutral': '😐', 
    'sad': '😢', 
    'surprise': '😲'
}

# Create transform based on settings
if USE_GRAYSCALE:
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN * 3, std=NORMALIZE_STD * 3)
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN * 3, std=NORMALIZE_STD * 3)
    ])

# Hugging Face model info
HF_USERNAME = "hiral20"
HF_MODEL_NAME = "emotion-model"
HF_MODEL_FILE = "emotion_model.pth"
HUGGINGFACE_URL = f"https://huggingface.co/{HF_USERNAME}/{HF_MODEL_NAME}/resolve/main/{HF_MODEL_FILE}"

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("😊 Emotion Detection")
    st.markdown("---")
    st.markdown("### Model Configuration")
    st.markdown(f"**Input Size:** {INPUT_SIZE}x{INPUT_SIZE}")
    st.markdown(f"**Grayscale:** {USE_GRAYSCALE}")
    st.markdown(f"**Classes:** {len(CLASSES)}")
    st.markdown("---")
    st.markdown("### Supported Emotions")
    for emotion in CLASSES:
        st.write(f"{EMOJI_MAP.get(emotion, '❓')} {emotion.capitalize()}")
    st.markdown("---")
    st.markdown(f"**Model Source:** `{HF_USERNAME}/{HF_MODEL_NAME}`")

# -----------------------------
# HEADER
# -----------------------------
st.title("😊 Emotion Detection + Explainable AI")
st.markdown("### Real-time emotion recognition with AI explainability")
st.markdown("---")

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model(model_path):
    """Load the trained emotion detection model"""
    if not os.path.exists(model_path):
        return None
    
    try:
        # Load state dict to detect number of classes
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        
        if 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
            st.sidebar.info(f"Model has {num_classes} output classes")
        
        # Create model
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes if 'fc.weight' in state_dict else len(CLASSES))
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def download_model():
    """Download model from Hugging Face"""
    model_path = "emotion_model.pth"
    
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is not None:
            return model
    
    with st.spinner("Downloading model from Hugging Face..."):
        try:
            response = requests.get(HUGGINGFACE_URL, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            st.success("✅ Model downloaded!")
            return load_model(model_path)
            
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            
            # Manual upload fallback
            uploaded = st.file_uploader("Upload model manually", type=['pth', 'pt'])
            if uploaded:
                with open(model_path, 'wb') as f:
                    f.write(uploaded.getbuffer())
                return load_model(model_path)
            
            return None

# -----------------------------
# FACE DETECTION
# -----------------------------
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# GRAD-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handle_forward = target_layer.register_forward_hook(self.save_activation)
        self.handle_backward = target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (INPUT_SIZE, INPUT_SIZE))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    
    def remove_hooks(self):
        self.handle_forward.remove()
        self.handle_backward.remove()

def get_target_layer(model):
    try:
        return model.layer4[-1].conv2
    except:
        return model.layer4

# -----------------------------
# PREDICTION
# -----------------------------
def predict_emotion(model, face_image):
    try:
        if len(face_image.shape) == 3:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(face_rgb)
        input_tensor = transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        return prediction.item(), confidence.item(), input_tensor, probabilities[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

# -----------------------------
# VISUALIZATION
# -----------------------------
def draw_emotion_box(image, x, y, w, h, emotion, confidence):
    color = (0, 255, 0)
    label = f"{EMOJI_MAP.get(emotion, '❓')} {emotion.upper()} ({confidence*100:.1f}%)"
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image

def create_gradcam_overlay(face_image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
    return cv2.addWeighted(face_image, 1-alpha, heatmap, alpha, 0)

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    # Load models
    face_cascade = load_face_cascade()
    model = download_model()
    
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face link.")
        st.stop()
    
    # Tabs
    tab1, tab2 = st.tabs(["📷 Image Upload", "ℹ️ About"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload Images", 
            type=['jpg', 'jpeg', 'png'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                st.markdown("---")
                
                # Read image
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    st.error("Could not read image")
                    continue
                
                # Detect faces
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    st.warning("No faces detected")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    continue
                
                result_image = image.copy()
                
                # Process each face
                for (x, y, w, h) in faces:
                    face = image[y:y+h, x:x+w]
                    pred_idx, confidence, input_tensor, probabilities = predict_emotion(model, face)
                    
                    if pred_idx is not None and pred_idx < len(CLASSES):
                        emotion = CLASSES[pred_idx]
                        
                        # Generate Grad-CAM
                        target_layer = get_target_layer(model)
                        gradcam = GradCAM(model, target_layer)
                        cam = gradcam.generate(input_tensor, pred_idx)
                        gradcam.remove_hooks()
                        
                        # Create overlay
                        overlay = create_gradcam_overlay(face, cam)
                        result_image = draw_emotion_box(result_image, x, y, w, h, emotion, confidence)
                        
                        # Display
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Detected Face")
                        with col2:
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM Explanation")
                
                # Show result
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Detection Result")
    
    with tab2:
        st.markdown("## How to Fix Wrong Predictions")
        st.markdown("""
        If the model is predicting wrong emotions, you need to **match the preprocessing** used during training.
        
        ### Steps to fix:
        
        1. **Find out how your model was trained**
        2. **Update the constants at the top of app.py**
        3. **Redeploy the app**
        
        ### Current Configuration:
        """)
        st.code(f"""
INPUT_SIZE = {INPUT_SIZE}
USE_GRAYSCALE = {USE_GRAYSCALE}
NORMALIZE_MEAN = {NORMALIZE_MEAN}
NORMALIZE_STD = {NORMALIZE_STD}
CLASSES = {CLASSES}
        """)
        
        st.markdown("""
        ### Common Configurations:
        
        - **FER2013 dataset:** INPUT_SIZE=48, grayscale=True, mean=[0.5], std=[0.5]
        - **CK+ dataset:** INPUT_SIZE=48 or 96, grayscale=True, mean=[0.5], std=[0.5]
        - **AffectNet:** INPUT_SIZE=224, grayscale=False, ImageNet normalization
        
        ### Need help?
        Tell me how your model was trained and I'll give you the correct settings.
        """)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()
