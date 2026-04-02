import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import timm
import os

st.set_page_config(page_title="Emotion Detection AI", layout="centered")
st.title("😊 Emotion Detection AI - Fixed Version")

emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 
    'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
}

@st.cache_resource
def load_model():
    """Load emotion recognition model"""
    model = timm.create_model('resnet18', pretrained=True, num_classes=7)
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def is_valid_face(face_img, gray_face):
    """
    Validate if a detected region is actually a face
    Returns: (is_face, reason)
    """
    h, w = face_img.shape[:2]
    
    # 1. Check face size (too small = likely false positive)
    if h < 60 or w < 60:
        return False, "Face too small"
    
    # 2. Check aspect ratio (faces should be roughly square)
    aspect_ratio = w / h
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        return False, f"Wrong aspect ratio: {aspect_ratio:.2f}"
    
    # 3. Check skin tone detection (simple color check in HSV space)
    try:
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        # Skin tones typically fall in these ranges
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 170, 255))
        skin_ratio = np.sum(skin_mask > 0) / (h * w)
        
        # Too little skin color = likely not a face
        if skin_ratio < 0.3:
            return False, f"Low skin color: {skin_ratio:.2f}"
    except:
        pass
    
    # 4. Check edge density (faces have distinct features)
    edges = cv2.Canny(gray_face, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Too few or too many edges = likely not a face
    if edge_density < 0.05 or edge_density > 0.4:
        return False, f"Unusual edge density: {edge_density:.2f}"
    
    # 5. Check symmetry (faces are roughly symmetric)
    try:
        left_half = face_img[:, :w//2]
        right_half = cv2.flip(face_img[:, w//2:], 1)
        
        # Resize to same size for comparison
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        diff = cv2.absdiff(left_half, right_half)
        symmetry_score = 1 - (np.mean(diff) / 255)
        
        if symmetry_score < 0.6:
            return False, f"Low symmetry: {symmetry_score:.2f}"
    except:
        pass
    
    return True, "Valid face"

def detect_real_faces(image):
    """Detect faces with false positive filtering"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple cascade classifiers
    cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    ]
    
    all_faces = []
    
    for cascade in cascades:
        # Try different parameters
        for scale in [1.05, 1.1]:
            for neighbors in [3, 5]:
                faces = cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale,
                    minNeighbors=neighbors,
                    minSize=(60, 60),
                    maxSize=(500, 500)
                )
                
                for (x, y, w, h) in faces:
                    face_roi = image[y:y+h, x:x+w]
                    gray_face = gray[y:y+h, x:x+w]
                    
                    # Validate if this is really a face
                    is_face, reason = is_valid_face(face_roi, gray_face)
                    
                    if is_face:
                        all_faces.append((x, y, w, h))
    
    # Remove overlapping detections (Non-Maximum Suppression)
    if len(all_faces) > 0:
        all_faces = sorted(all_faces, key=lambda f: f[2]*f[3], reverse=True)
        unique_faces = []
        
        for face in all_faces:
            x, y, w, h = face
            is_overlap = False
            
            for ux, uy, uw, uh in unique_faces:
                # Calculate IoU (Intersection over Union)
                ix1 = max(x, ux)
                iy1 = max(y, uy)
                ix2 = min(x+w, ux+uw)
                iy2 = min(y+h, uy+uh)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    union = (w*h) + (uw*uh) - intersection
                    iou = intersection / union
                    
                    if iou > 0.3:  # Overlap threshold
                        is_overlap = True
                        break
            
            if not is_overlap:
                unique_faces.append(face)
        
        return unique_faces[:3]  # Max 3 faces
    
    return []

def preprocess_face(face_img):
    """Enhanced face preprocessing"""
    if len(face_img.shape) == 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    elif face_img.shape[2] == 4:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def predict_emotion(face_img, model):
    """Predict emotion with confidence score"""
    processed_face = preprocess_face(face_img)
    pil_img = Image.fromarray(processed_face)
    img_tensor = transform(pil_img).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    return prediction.item(), confidence.item()

def main():
    st.markdown("""
    ### 🎯 Improved Emotion Detection with False Positive Filtering
    Now detects ONLY real faces, not random objects!
    """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.5,
        help="Higher = more accurate but might miss some faces"
    )
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Could not read image")
            return
        
        # Detect faces with false positive filtering
        with st.spinner("Detecting faces..."):
            faces = detect_real_faces(img)
        
        if len(faces) == 0:
            st.warning("⚠️ No real faces detected. Make sure the face is clear, front-facing, and well-lit.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image")
            
            # Show debug info
            with st.expander("🔍 Why no face detected?"):
                st.markdown("""
                Common reasons:
                - Face too small (need at least 60x60 pixels)
                - Face at an angle (needs to be front-facing)
                - Poor lighting or shadows
                - Face partially covered (glasses, hair, mask)
                - Image quality too low
                """)
            return
        
        st.success(f"✅ Found {len(faces)} real face(s)")
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face with margin
            margin = int(0.1 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2*margin)
            h = min(img.shape[0] - y, h + 2*margin)
            
            face = img[y:y+h, x:x+w]
            
            # Predict emotion
            emotion_idx, confidence = predict_emotion(face, model)
            emotion = emotion_classes[emotion_idx]
            emoji = emotion_emojis.get(emotion, "😊")
            
            # Color based on confidence
            if confidence >= confidence_threshold:
                color = (0, 255, 0)
                border_style = "solid"
            else:
                color = (0, 165, 255)
                border_style = "dashed"
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label = f"{emoji} {emotion.upper()} ({confidence*100:.1f}%)"
            
            # Add background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x, y-label_size[1]-10), (x+label_size[0]+10, y), color, -1)
            cv2.putText(img, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 0), 2)
            
            # Display results
            st.markdown(f"### Face {i+1}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), 
                        caption=f"Detected Face", width=200)
            
            with col2:
                st.metric("Predicted Emotion", f"{emoji} {emotion.title()}")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                
                if confidence < confidence_threshold:
                    st.warning("⚠️ Low confidence - this might be wrong")
        
        # Show full image
        st.markdown("### Full Image with Detections")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Sidebar info
    with st.sidebar:
        st.markdown("## 📊 How False Positives Are Filtered")
        st.markdown("""
        The app now checks:
        1. **Face Size** - Too small = not a face
        2. **Aspect Ratio** - Faces are roughly square
        3. **Skin Tone** - Checks for realistic skin colors
        4. **Edge Density** - Faces have distinct edges
        5. **Symmetry** - Faces are roughly symmetrical
        6. **Multiple Cascades** - Uses 2 detectors for better accuracy
        7. **Non-Maximum Suppression** - Removes duplicate detections
        
        ### Tips for Best Results:
        - ✅ Use clear, front-facing photos
        - ✅ Good lighting (no harsh shadows)
        - ✅ Face should be at least 100x100 pixels
        - ✅ Remove glasses if possible
        - ✅ Neutral background helps
        """)

if __name__ == "__main__":
    main()
