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
import matplotlib.pyplot as plt

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
# Default classes - will be updated based on model
DEFAULT_CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASSES = DEFAULT_CLASSES.copy()

EMOJI_MAP = {
    'angry': '😠',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲',
    'disgust': '🤢',
    'contempt': '😏'
}

COLORS = {
    'angry': (0, 0, 255),
    'fear': (255, 0, 255),
    'happy': (0, 255, 0),
    'neutral': (255, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (0, 255, 255),
    'disgust': (0, 128, 128),
    'contempt': (128, 0, 128)
}

# Hugging Face model info
HF_USERNAME = "hiral20"
HF_MODEL_NAME = "emotion-model"
HF_MODEL_FILE = "emotion_model.pth"
HUGGINGFACE_URL = f"https://huggingface.co/{HF_USERNAME}/{HF_MODEL_NAME}/resolve/main/{HF_MODEL_FILE}"

# -----------------------------
# PREPROCESSING OPTIONS
# -----------------------------
preprocessing_options = {
    "Option 1: 96x96 with [-1,1] Norm": transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "Option 2: 96x96 No Normalization": transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]),
    "Option 3: 96x96 ImageNet Norm": transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "Option 4: 48x48 with [-1,1] Norm (FER2013 style)": transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "Option 5: 48x48 No Normalization": transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]),
    "Option 6: 224x224 ImageNet Norm": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
    for emotion in DEFAULT_CLASSES:
        st.write(f"{EMOJI_MAP.get(emotion, '❓')} {emotion.capitalize()}")
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown(f"**Source:** Hugging Face")
    st.markdown(f"**Repo:** `{HF_USERNAME}/{HF_MODEL_NAME}`")
    st.markdown(f"**File:** `{HF_MODEL_FILE}`")
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
# MODEL LOADING WITH FLEXIBLE ARCHITECTURE
# -----------------------------
@st.cache_resource
def load_model(model_path):
    """Load the trained emotion detection model with flexible architecture"""
    if not os.path.exists(model_path):
        return None, DEFAULT_CLASSES
    
    try:
        # Load state dict first to inspect
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        
        # Detect number of classes from state_dict
        num_classes = DEFAULT_CLASSES
        if 'fc.weight' in state_dict:
            output_features = state_dict['fc.weight'].shape[0]
            st.info(f"📊 Model detected: {output_features} output classes")
            
            # Map common class counts
            if output_features == 6:
                class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            elif output_features == 7:
                class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            elif output_features == 8:
                class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'contempt']
            else:
                class_names = [f'class_{i}' for i in range(output_features)]
            
            # Create model with correct number of classes
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, output_features)
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.eval()
            
            return model, class_names
        else:
            st.error("Could not detect model architecture")
            return None, DEFAULT_CLASSES
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, DEFAULT_CLASSES

def download_model_from_huggingface(model_path):
    """Download model from Hugging Face with progress bar"""
    try:
        st.info(f"📥 Downloading model from Hugging Face...")
        st.code(f"Source: {HUGGINGFACE_URL}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(HUGGINGFACE_URL, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
                else:
                    status_text.text(f"Downloading: {downloaded/(1024*1024):.1f}MB downloaded")
        
        progress_bar.empty()
        status_text.empty()
        
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            st.success(f"✅ Model downloaded successfully! ({os.path.getsize(model_path)/(1024*1024):.1f}MB)")
            time.sleep(1)
            return True
        else:
            return False
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

# -----------------------------
# MODEL FILE HANDLER
# -----------------------------
def handle_model_loading():
    """Handle model file download from Hugging Face"""
    model_path = "emotion_model.pth"
    
    if os.path.exists(model_path):
        test_model, class_names = load_model(model_path)
        if test_model is not None:
            return test_model, class_names
        else:
            st.warning("Existing model file is corrupted. Re-downloading...")
            os.remove(model_path)
    
    st.warning("⚠️ Model file not found. Downloading from Hugging Face...")
    
    with st.spinner("Connecting to Hugging Face..."):
        success = download_model_from_huggingface(model_path)
    
    if not success:
        st.markdown("---")
        st.markdown("### 📁 Manual Upload (Alternative)")
        
        uploaded_file = st.file_uploader(
            "Upload emotion_model.pth",
            type=['pth', 'pt'],
            key="manual_upload"
        )
        
        if uploaded_file is not None:
            with open(model_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Model uploaded successfully!")
            time.sleep(1)
            st.rerun()
        
        return None, DEFAULT_CLASSES
    
    model, class_names = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model after download")
        return None, DEFAULT_CLASSES
    
    st.success("✅ Model ready!")
    time.sleep(1)
    return model, class_names

# -----------------------------
# FACE DETECTION
# -----------------------------
@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# -----------------------------
# GRAD-CAM IMPLEMENTATION
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
        cam = cv2.resize(cam, (96, 96))
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def remove_hooks(self):
        self.handle_forward.remove()
        self.handle_backward.remove()

def get_target_layer(model):
    try:
        return model.layer4[-1].conv2
    except AttributeError:
        try:
            return model.layer4[1].conv2
        except:
            return model.layer4

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_emotion(model, face_image, transform_func, class_names):
    """Predict emotion from face image with specified transform"""
    try:
        if len(face_image.shape) == 3:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(face_rgb)
        input_tensor = transform_func(pil_image).unsqueeze(0)
        
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
def draw_emotion_box(image, x, y, w, h, emotion, confidence, class_names):
    color = COLORS.get(emotion, (0, 255, 0))
    
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    label = f"{EMOJI_MAP.get(emotion, '❓')} {emotion.upper()} ({confidence*100:.1f}%)"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    cv2.rectangle(image, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
    cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image

def create_gradcam_overlay(face_image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
    overlay = cv2.addWeighted(face_image, 1-alpha, heatmap, alpha, 0)
    return overlay

def plot_confidence_chart(probabilities, class_names):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    probs_np = probabilities.detach().cpu().numpy()
    colors_list = [COLORS.get(cls, (100, 100, 100)) for cls in class_names[:len(probs_np)]]
    colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors_list]
    
    bars = ax.bar(class_names[:len(probs_np)], probs_np, color=colors_rgb, alpha=0.7)
    ax.set_ylabel('Confidence Score')
    ax.set_title('Emotion Probabilities')
    ax.set_ylim([0, 1])
    
    for bar, prob in zip(bars, probs_np):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def display_model_info(model, class_names):
    """Display model information for debugging"""
    st.sidebar.markdown("### 📊 Model Info")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    st.sidebar.markdown(f"**Total Parameters:** {total_params:,}")
    st.sidebar.markdown(f"**Trainable:** {trainable_params:,}")
    
    if hasattr(model, 'fc'):
        output_features = model.fc.out_features
        st.sidebar.markdown(f"**Output Classes:** {output_features}")
        st.sidebar.markdown(f"**Class Names:** {', '.join(class_names[:output_features])}")

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    # Load face cascade
    face_cascade = load_face_cascade()
    
    # Handle model loading
    model, class_names = handle_model_loading()
    
    if model is None:
        st.stop()
    
    # Update global CLASSES
    global CLASSES
    CLASSES = class_names
    
    # Display model info in sidebar
    display_model_info(model, CLASSES)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Upload", "🎥 Webcam", "🔧 Debug Mode", "ℹ️ About"])
    
    # -------------------------
    # TAB 1: IMAGE UPLOAD
    # -------------------------
    with tab1:
        st.subheader("Upload Images for Emotion Detection")
        
        # Preprocessing selector
        st.markdown("### ⚙️ Preprocessing Settings")
        st.info("💡 If predictions are wrong, try different preprocessing options below")
        
        preprocessing_choice = st.selectbox(
            "Select Preprocessing Method",
            list(preprocessing_options.keys()),
            key="preprocess_main"
        )
        current_transform = preprocessing_options[preprocessing_choice]
        
        st.caption(f"Currently using: **{preprocessing_choice}**")
        
        uploaded_files = st.file_uploader(
            "Choose images (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for idx, file in enumerate(uploaded_files):
                st.markdown("---")
                st.subheader(f"Image {idx + 1}: {file.name}")
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    st.error("Could not read image")
                    continue
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    st.warning("No faces detected in this image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    continue
                
                result_image = image.copy()
                cols = st.columns(min(len(faces), 3))
                
                for i, (x, y, w, h) in enumerate(faces):
                    face = image[y:y+h, x:x+w]
                    
                    pred_idx, confidence, input_tensor, probabilities = predict_emotion(
                        model, face, current_transform, CLASSES
                    )
                    
                    if pred_idx is not None and pred_idx < len(CLASSES):
                        emotion = CLASSES[pred_idx]
                        
                        target_layer = get_target_layer(model)
                        gradcam = GradCAM(model, target_layer)
                        cam = gradcam.generate(input_tensor, pred_idx)
                        gradcam.remove_hooks()
                        
                        overlay = create_gradcam_overlay(face, cam, alpha=0.4)
                        result_image = draw_emotion_box(result_image, x, y, w, h, emotion, confidence, CLASSES)
                        
                        with cols[i % len(cols)]:
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                    caption=f"{EMOJI_MAP.get(emotion, '❓')} {emotion.capitalize()} ({confidence*100:.1f}%)",
                                    use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Detected Emotions", use_container_width=True)
                
                if len(faces) > 0 and 'probabilities' in locals() and probabilities is not None:
                    st.subheader("📊 Confidence Analysis")
                    fig = plot_confidence_chart(probabilities, CLASSES)
                    st.pyplot(fig)
    
    # -------------------------
    # TAB 2: WEBCAM
    # -------------------------
    with tab2:
        st.subheader("Real-time Emotion Detection")
        st.warning("⚠️ Note: Webcam mode works best when running locally.")
        
        preprocessing_choice_webcam = st.selectbox(
            "Select Preprocessing Method for Webcam",
            list(preprocessing_options.keys()),
            key="preprocess_webcam"
        )
        webcam_transform = preprocessing_options[preprocessing_choice_webcam]
        
        run_webcam = st.checkbox("Start Webcam", key="webcam_checkbox")
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access webcam.")
            else:
                frame_placeholder = st.empty()
                stop_button = st.button("Stop Webcam")
                
                while run_webcam and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        face = frame[y:y+h, x:x+w]
                        pred_idx, confidence, _, _ = predict_emotion(
                            model, face, webcam_transform, CLASSES
                        )
                        
                        if pred_idx is not None and pred_idx < len(CLASSES):
                            emotion = CLASSES[pred_idx]
                            frame = draw_emotion_box(frame, x, y, w, h, emotion, confidence, CLASSES)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    if stop_button:
                        break
                
                cap.release()
                st.success("Webcam stopped")
    
    # -------------------------
    # TAB 3: DEBUG MODE
    # -------------------------
    with tab3:
        st.subheader("🔧 Debug Mode")
        st.markdown("Use this section to test different preprocessing options and see raw model outputs")
        
        debug_file = st.file_uploader(
            "Upload a test image for debugging",
            type=['jpg', 'jpeg', 'png'],
            key="debug_upload"
        )
        
        if debug_file is not None:
            file_bytes = np.asarray(bytearray(debug_file.read()), dtype=np.uint8)
            debug_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if debug_image is not None:
                gray = cv2.cvtColor(debug_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    st.error("No face detected in the uploaded image")
                else:
                    st.success(f"Found {len(faces)} face(s)")
                    
                    for idx, (x, y, w, h) in enumerate(faces):
                        st.markdown(f"### Face {idx + 1}")
                        face = debug_image[y:y+h, x:x+w]
                        st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), width=200)
                        
                        # Test all preprocessing options
                        st.markdown("#### Predictions with different preprocessing:")
                        
                        results = []
                        for opt_name, transform_func in preprocessing_options.items():
                            pred_idx, confidence, _, probs = predict_emotion(
                                model, face, transform_func, CLASSES
                            )
                            if pred_idx is not None and pred_idx < len(CLASSES):
                                results.append({
                                    'method': opt_name,
                                    'emotion': CLASSES[pred_idx],
                                    'confidence': confidence,
                                    'probabilities': probs
                                })
                        
                        # Display results in a table
                        for result in results:
                            emoji = EMOJI_MAP.get(result['emotion'], '❓')
                            st.write(f"**{result['method']}:** {emoji} {result['emotion']} ({result['confidence']*100:.1f}%)")
                        
                        # Show full probability distribution for the best method
                        st.markdown("#### Full Probability Distribution (using Option 4 - FER2013 style):")
                        best_result = max(results, key=lambda x: x['confidence']) if results else None
                        if best_result:
                            fig = plot_confidence_chart(best_result['probabilities'], CLASSES)
                            st.pyplot(fig)
                            
                            # Raw values
                            with st.expander("Show Raw Probability Values"):
                                probs_np = best_result['probabilities'].detach().cpu().numpy()
                                for i, (cls, prob) in enumerate(zip(CLASSES[:len(probs_np)], probs_np)):
                                    st.write(f"{cls}: {prob:.4f}")
                
                st.markdown("---")
                st.markdown("### 💡 Troubleshooting Tips")
                st.markdown("""
                1. **If predictions are consistently wrong**, try the preprocessing method that gives the highest confidence
                2. **Check if the model was trained on a different dataset** (FER2013 vs CK+ vs AffectNet)
                3. **Ensure faces are front-facing and well-lit**
                4. **The model might expect grayscale or RGB images** - our preprocessing handles both
                5. **Class ordering might be different** - check the raw probability values above
                """)
    
    # -------------------------
    # TAB 4: ABOUT
    # -------------------------
    with tab4:
        st.markdown(f"""
        ## About This Application
        
        ### 🎯 Purpose
        Emotion detection with explainable AI using Grad-CAM visualizations.
        
        ### 🧠 Model Details
        - **Architecture**: ResNet18
        - **Source**: Hugging Face - `{HF_USERNAME}/{HF_MODEL_NAME}`
        - **File**: `{HF_MODEL_FILE}`
        - **Detected Classes**: {len(CLASSES)} emotions
        
        ### 🔧 How to Fix Wrong Predictions
        
        If the model is predicting wrong emotions:
        
        1. **Try different preprocessing options** in the Image Upload tab
        2. **Use Debug Mode** to test all preprocessing methods at once
        3. **Check the raw probability values** to see what the model is actually outputting
        
        ### 📊 Common Preprocessing Methods
        
        - **Option 4 (48x48 FER2013 style)** - Most common for emotion detection
        - **Option 1 (96x96 with normalization)** - Good for custom datasets
        - **Option 6 (224x224 ImageNet)** - If model was fine-tuned from ImageNet
        
        ### 🔧 Technical Stack
        - Streamlit for UI
        - PyTorch for deep learning
        - OpenCV for face detection
        - Grad-CAM for explainability
        
        ### 📥 Model Download
        The model is automatically downloaded from:
        `{HUGGINGFACE_URL}`
        """)

# -----------------------------
# RUN APPLICATION
# -----------------------------
if __name__ == "__main__":
    main()
