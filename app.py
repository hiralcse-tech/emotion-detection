import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path
import requests
import tempfile

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection with Explainable AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CONSTANTS
# -----------------------------
CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOJI_MAP = {
    'angry': '😠',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}
COLORS = {
    'angry': (0, 0, 255),      # Red
    'fear': (255, 0, 255),     # Purple
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Cyan
    'sad': (255, 0, 0),        # Blue
    'surprise': (0, 255, 255)  # Yellow
}

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("😊 Emotion Detection")
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This app uses Deep Learning to detect emotions from facial expressions "
        "and explains its decisions using Grad-CAM visualization."
    )
    st.markdown("### Supported Emotions")
    for emotion, emoji in EMOJI_MAP.items():
        st.write(f"{emoji} {emotion.capitalize()}")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Face Detection**: Haar Cascade Classifier\n"
        "2. **Emotion Recognition**: ResNet18 Neural Network\n"
        "3. **Explainability**: Grad-CAM heatmaps"
    )
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# -----------------------------
# HEADER
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("😊 Emotion Detection + Explainable AI")
    st.markdown("### Real-time emotion recognition with AI explainability")
st.markdown("---")

# -----------------------------
# MODEL LOADING (WITHOUT WIDGETS IN CACHED FUNCTION)
# -----------------------------
@st.cache_resource
def load_model(model_path):
    """Load the trained emotion detection model"""
    if not os.path.exists(model_path):
        return None
    
    try:
        # Create model architecture
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        
        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# -----------------------------
# FACE DETECTION
# -----------------------------
@st.cache_resource
def load_face_cascade():
    """Load Haar Cascade for face detection"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# -----------------------------
# IMAGE TRANSFORMATIONS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -----------------------------
# GRAD-CAM IMPLEMENTATION
# -----------------------------
class GradCAM:
    """GradCAM for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.handle_forward = target_layer.register_forward_hook(self.save_activation)
        self.handle_backward = target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_tensor, target_class):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass
        loss = output[0, target_class]
        loss.backward()
        
        # Process gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (96, 96))
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def remove_hooks(self):
        self.handle_forward.remove()
        self.handle_backward.remove()

def get_target_layer(model):
    """Get appropriate target layer for Grad-CAM"""
    try:
        # Try to get the last convolutional layer of ResNet18
        return model.layer4[-1].conv2
    except AttributeError:
        try:
            return model.layer4[1].conv2
        except:
            return model.layer4

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_emotion(model, face_image):
    """Predict emotion from face image"""
    try:
        # Convert to PIL Image
        if len(face_image.shape) == 3:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(face_rgb)
        
        # Transform
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        return prediction.item(), confidence.item(), input_tensor, probabilities[0]
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

# -----------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------
def draw_emotion_box(image, x, y, w, h, emotion, confidence):
    """Draw bounding box with emotion label"""
    color = COLORS.get(emotion, (0, 255, 0))
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    # Draw label background
    label = f"{EMOJI_MAP.get(emotion, '')} {emotion.upper()} ({confidence*100:.1f}%)"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    cv2.rectangle(image, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
    cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image

def create_gradcam_overlay(face_image, cam, alpha=0.5):
    """Create Grad-CAM heatmap overlay"""
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
    
    # Overlay
    overlay = cv2.addWeighted(face_image, 1-alpha, heatmap, alpha, 0)
    
    return overlay

def plot_confidence_chart(probabilities):
    """Plot confidence scores for all emotions"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = [COLORS[cls] for cls in CLASSES]
    colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_list]
    
    bars = ax.bar(CLASSES, probabilities.detach().cpu().numpy(), color=colors_rgb, alpha=0.7)
    ax.set_ylabel('Confidence Score')
    ax.set_title('Emotion Probabilities')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# -----------------------------
# MODEL FILE HANDLER (Widgets outside cached function)
# -----------------------------
def handle_model_loading():
    """Handle model file upload and loading (widgets here, outside cache)"""
    
    # Check if model file exists
    model_path = "emotion_model.pth"
    
    if not os.path.exists(model_path):
        st.warning("⚠️ Model file not found!")
        
        # Option 1: Upload model file
        st.markdown("### Upload Your Model File")
        uploaded_file = st.file_uploader(
            "Upload your trained model file (emotion_model.pth)",
            type=['pth', 'pt'],
            help="Upload the PyTorch model file you trained"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with open(model_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Model uploaded successfully! Please refresh or continue.")
            st.rerun()
        
        st.markdown("---")
        st.info("""
        **Need a model?** 
        Train your own model using ResNet18 on a dataset like FER2013 or CK+.
        The model should have 6 output classes: angry, fear, happy, neutral, sad, surprise.
        """)
        
        return None
    
    # Load the model
    with st.spinner("Loading emotion detection model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        # Option to delete corrupted model
        if st.button("🗑️ Delete corrupted model file"):
            os.remove(model_path)
            st.rerun()
        return None
    
    return model

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    # Load face cascade
    face_cascade = load_face_cascade()
    
    # Handle model loading (widgets are here, not in cached function)
    model = handle_model_loading()
    
    if model is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Webcam", "ℹ️ About"])
    
    # -------------------------
    # TAB 1: IMAGE UPLOAD
    # -------------------------
    with tab1:
        st.subheader("Upload Images for Emotion Detection")
        
        uploaded_files = st.file_uploader(
            "Choose images (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for idx, file in enumerate(uploaded_files):
                st.markdown("---")
                st.subheader(f"Image {idx + 1}: {file.name}")
                
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
                    st.warning("No faces detected in this image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    continue
                
                # Process each face
                result_image = image.copy()
                cols = st.columns(min(len(faces), 3))
                
                for i, (x, y, w, h) in enumerate(faces):
                    face = image[y:y+h, x:x+w]
                    
                    # Predict emotion
                    pred_idx, confidence, input_tensor, probabilities = predict_emotion(model, face)
                    
                    if pred_idx is not None:
                        emotion = CLASSES[pred_idx]
                        
                        # Generate Grad-CAM
                        target_layer = get_target_layer(model)
                        gradcam = GradCAM(model, target_layer)
                        cam = gradcam.generate(input_tensor, pred_idx)
                        gradcam.remove_hooks()
                        
                        # Create overlays
                        overlay = create_gradcam_overlay(face, cam, alpha=0.4)
                        
                        # Draw on result image
                        result_image = draw_emotion_box(result_image, x, y, w, h, emotion, confidence)
                        
                        # Display in columns
                        with cols[i % len(cols)]:
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                    caption=f"{EMOJI_MAP[emotion]} {emotion.capitalize()} ({confidence*100:.1f}%)",
                                    use_container_width=True)
                
                # Show original image with boxes
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            caption="Original Image", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                            caption="Detected Emotions", use_container_width=True)
                
                # Show confidence chart for first face
                if len(faces) > 0 and 'probabilities' in locals():
                    st.subheader("Confidence Analysis")
                    fig = plot_confidence_chart(probabilities)
                    st.pyplot(fig)
    
    # -------------------------
    # TAB 2: WEBCAM
    # -------------------------
    with tab2:
        st.subheader("Real-time Emotion Detection")
        st.warning("⚠️ Note: Webcam mode works best when running locally. Streamlit Cloud may have limited webcam support.")
        
        run_webcam = st.checkbox("Start Webcam", key="webcam_checkbox")
        
        if run_webcam:
            # Try to open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access webcam. Please check:")
                st.markdown("""
                - Camera permissions are granted
                - No other app is using the camera
                - Camera is properly connected
                """)
            else:
                frame_placeholder = st.empty()
                stop_button = st.button("Stop Webcam")
                
                while run_webcam and not stop_button:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Detect faces
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    # Process faces
                    for (x, y, w, h) in faces:
                        face = frame[y:y+h, x:x+w]
                        pred_idx, confidence, _, _ = predict_emotion(model, face)
                        
                        if pred_idx is not None:
                            emotion = CLASSES[pred_idx]
                            frame = draw_emotion_box(frame, x, y, w, h, emotion, confidence)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                cap.release()
                st.success("Webcam stopped")
    
    # -------------------------
    # TAB 3: ABOUT
    # -------------------------
    with tab3:
        st.markdown("""
        ## About This Application
        
        ### 🎯 Purpose
        This application demonstrates how deep learning can be used for emotion recognition while providing explainability through Grad-CAM visualizations.
        
        ### 🧠 How It Works
        
        #### 1. Face Detection
        - Uses OpenCV's Haar Cascade Classifier
        - Detects frontal faces in images/video streams
        
        #### 2. Emotion Recognition
        - **Model Architecture**: ResNet18 (pre-trained on ImageNet)
        - **Fine-tuned**: Modified final layer for 6 emotion classes
        - **Input Size**: 96x96 pixels
        - **Classes**: Angry, Fear, Happy, Neutral, Sad, Surprise
        
        #### 3. Explainable AI (Grad-CAM)
        - **Gradient-weighted Class Activation Mapping**
        - Highlights important regions the model focuses on
        - Red regions indicate high importance for prediction
        - Helps understand model decision-making
        
        ### 📊 Model Training
        - Typically trained on datasets like FER2013 or CK+
        - Transfer learning from ImageNet weights
        - Optimized for real-time performance
        
        ### 🔧 Technical Stack
        - **Frontend**: Streamlit
        - **Deep Learning**: PyTorch
        - **Computer Vision**: OpenCV, torchvision
        - **Visualization**: Matplotlib
        
        ### 📝 Requirements
        - Python 3.8+
        - PyTorch 1.9+
        - OpenCV
        - Streamlit
        """)

# -----------------------------
# RUN APPLICATION
# -----------------------------
if __name__ == "__main__":
    main()
