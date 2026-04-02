import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import os
import requests

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("😊 Emotion Detection AI")

classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "emotion_model.pth"

# Emotion descriptions for better UX
emotion_descriptions = {
    'angry': "😠 Showing signs of anger or frustration",
    'fear': "😨 Appears scared or anxious",
    'happy': "😊 Showing happiness or joy",
    'neutral': "😐 No strong emotion detected",
    'sad': "😢 Appears sad or down",
    'surprise': "😲 Showing surprise or shock"
}

@st.cache_resource
def load_model():
    url = "https://huggingface.co/hiral20/emotion-model/resolve/main/emotion_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI model... (first time only)"):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
                st.success("✅ Model downloaded!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                return None
    
    try:
        # Use pretrained weights as base
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 6)
        
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None

# Improved preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

transform = get_transform()

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def enhance_face(face_img):
    """Enhance face image for better prediction"""
    # Convert to RGB if needed
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
    elif face_img.shape[2] == 3 and face_img.dtype == np.uint8:
        # Already RGB/BGR, ensure BGR to RGB
        if not isinstance(face_img[0,0,0], np.floating):
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def predict_emotion(face_img, model):
    """Improved prediction with multiple augmentations"""
    # Enhance face
    enhanced_face = enhance_face(face_img)
    
    # Try multiple slightly different versions and average
    predictions = []
    
    # Original
    pil_img = Image.fromarray(enhanced_face)
    img_tensor = transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        predictions.append(probs.numpy()[0])
    
    # Horizontal flip (mirror image)
    flipped = cv2.flip(enhanced_face, 1)
    pil_flipped = Image.fromarray(flipped)
    img_flipped = transform(pil_flipped).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_flipped)
        probs = torch.softmax(output, dim=1)
        predictions.append(probs.numpy()[0])
    
    # Average predictions
    avg_probs = np.mean(predictions, axis=0)
    prediction = np.argmax(avg_probs)
    confidence = avg_probs[prediction]
    
    return prediction, confidence

def preprocess_face_detection(gray_img):
    """Improve face detection"""
    # Apply histogram equalization for better detection
    equalized = cv2.equalizeHist(gray_img)
    return equalized

def main():
    model = load_model()
    if model is None:
        st.stop()
    
    face_cascade = load_face_cascade()
    
    # Add confidence threshold slider
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Lower = more predictions but possibly wrong, Higher = fewer but more confident"
    )
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Could not read image")
            return
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_gray = preprocess_face_detection(gray)
        
        # Detect faces with different parameters
        faces = face_cascade.detectMultiScale(
            enhanced_gray, 
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # Less strict
            minSize=(48, 48),  # Smaller minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            # Try with different parameters
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            st.warning("⚠️ No face detected. Try an image with a clear, front-facing face.")
            st.image(img_rgb, caption="Uploaded Image")
            return
        
        st.success(f"✅ Found {len(faces)} face(s)")
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Add margin around face
            margin = int(0.1 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2*margin)
            h = min(img.shape[0] - y, h + 2*margin)
            
            # Extract and enhance face
            face = img[y:y+h, x:x+w]
            
            # Predict emotion
            emotion_idx, confidence = predict_emotion(face, model)
            emotion = classes[emotion_idx]
            
            # Only show if confidence meets threshold
            if confidence >= confidence_threshold:
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)  # Orange for low confidence
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion.upper()} ({confidence*100:.1f}%)"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Display results
            st.markdown(f"### Face {i+1}")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), 
                        caption=f"Detected Face", width=200)
            
            with col2:
                st.metric("Predicted Emotion", f"{emotion.title()}")
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with col3:
                st.info(emotion_descriptions.get(emotion, ""))
            
            # Show confidence distribution
            st.markdown("**Confidence Distribution:**")
            # Get probabilities for all emotions
            face_enhanced = enhance_face(face)
            pil_face = Image.fromarray(face_enhanced)
            face_tensor = transform(pil_face).unsqueeze(0)
            
            with torch.no_grad():
                output = model(face_tensor)
                probs = torch.softmax(output, dim=1).numpy()[0]
            
            # Create bar chart
            chart_data = {classes[i]: float(probs[i]) for i in range(len(classes))}
            st.bar_chart(chart_data)
        
        # Show full image
        st.markdown("### Full Image with Detections")
        img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_final)
    
    # Sidebar info
    with st.sidebar:
        st.markdown("## 📊 About")
        st.markdown("""
        ### Why predictions might be wrong:
        1. **Poor image quality** - blurry or dark images
        2. **Face angle** - profile or tilted faces
        3. **Lighting** - shadows or overexposure
        4. **Face accessories** - glasses, hats, masks
        5. **Resolution** - face too small
        
        ### Tips for better accuracy:
        - ✅ Use clear, well-lit photos
        - ✅ Face should be front-facing
        - ✅ Remove glasses if possible
        - ✅ Face should be at least 100x100 pixels
        - ✅ Neutral expression for neutral detection
        
        ### Current limitations:
        - Model trained on FER2013 (65-70% accuracy)
        - Works best on adult faces
        - May struggle with extreme expressions
        """)

if __name__ == "__main__":
    main()
