import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import cv2

# Model definition (must match training code)
class FaceAntiSpoofingNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.7):
        super().__init__()
        self.backbone = efficientnet_v2_s(pretrained=True)
        
        # Freeze more layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier with stronger regularization
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Load model function
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceAntiSpoofingNet(num_classes=2, dropout_rate=0.7)
    
    try:
        # Load the trained weights
        model.load_state_dict(torch.load('best_face_antispoofing_model.pth', map_location=device))
        model.to(device)  # Move model to device
        model.eval()
        st.success(f"✅ Model loaded successfully on {device}")
        return model, device
    except FileNotFoundError:
        st.error("❌ Model file 'best_face_antispoofing_model.pth' not found. Please train the model first.")
        return None, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, device

# Image preprocessing
def preprocess_image(image):
    # Auto-rotate to portrait if needed
    width, height = image.size
    if width > height:
        image = image.rotate(90, expand=True)
    
    # Resize large images to manageable size first
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

# Prediction function
def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        # Ensure both model and tensor are on the same device
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        
        outputs = model(image_tensor)
        
        # Get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities[0].cpu().numpy()

# Face detection function (optional enhancement)
def detect_faces(image):
    """Detect faces in the image using OpenCV"""
    # Convert PIL to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces, opencv_image

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Face Anti-Spoofing Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Face Anti-Spoofing Detector")
    st.markdown("Upload an image to detect if it's a **real face** or **spoof attack**")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a deep learning model to detect:
        - **Live faces** (real person)
        - **Spoof attacks** (photos, videos, masks)
        
        **Model Details:**
        - Architecture: EfficientNet-V2-S
        - Dataset: CASIA-FASD
        - Accuracy: ~96%
        """)
        
        st.header("📊 Class Labels")
        st.write("🟢 **Live (1)**: Real person")
        st.write("🔴 **Spoof (0)**: Attack attempt")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a face image to analyze"
        )
        
        # Camera input option
        st.subheader("📷 Or take a photo")
        camera_image = st.camera_input("Take a picture")
        
        # Use camera image if available, otherwise uploaded file
        image_source = camera_image if camera_image else uploaded_file
        
        if image_source:
            # Load and display image
            image = Image.open(image_source)
            
            # Auto-rotate and resize for display
            display_image = image.copy()
            width, height = display_image.size
            
            # Auto-rotate to portrait if landscape
            if width > height:
                display_image = display_image.rotate(90, expand=True)
                st.info("📱 Image auto-rotated to portrait orientation")
            
            # Resize for display if too large
            max_display_size = 800
            if max(display_image.size) > max_display_size:
                ratio = max_display_size / max(display_image.size)
                new_size = tuple(int(dim * ratio) for dim in display_image.size)
                display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)
                st.info(f"📐 High resolution image resized for display ({width}x{height} → {new_size[0]}x{new_size[1]})")
            
            st.image(display_image, caption="Input Image", use_container_width=True)
            
            # Optional face detection
            if st.checkbox("🔍 Show face detection", value=True):
                faces, opencv_image = detect_faces(display_image)
                if len(faces) > 0:
                    st.success(f"✅ {len(faces)} face(s) detected")
                    
                    # Draw rectangles around faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Convert back to RGB for display
                    opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                    st.image(opencv_image_rgb, caption="Face Detection", use_container_width=True)
                else:
                    st.warning("⚠️ No faces detected")
    
    with col2:
        st.header("🔮 Prediction Results")
        
        if image_source:
            # Make prediction button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Make prediction
                    prediction, confidence, probabilities = predict_image(model, image_tensor, device)
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Main prediction
                    if prediction == 1:
                        st.success(f"🟢 **LIVE FACE** (Confidence: {confidence:.1%})")
                        st.balloons()
                    else:
                        st.error(f"🔴 **SPOOF DETECTED** (Confidence: {confidence:.1%})")
                    
                    # Probability breakdown
                    st.subheader("📈 Probability Breakdown")
                    
                    col_spoof, col_live = st.columns(2)
                    with col_spoof:
                        st.metric("🔴 Spoof Probability", f"{probabilities[0]:.1%}")
                    with col_live:
                        st.metric("🟢 Live Probability", f"{probabilities[1]:.1%}")
                    
                    # Confidence bar
                    st.subheader("📊 Confidence Level")
                    if prediction == 1:
                        st.progress(float(probabilities[1]))
                    else:
                        st.progress(float(probabilities[0]))
                    
                    # Additional info
                    st.subheader("🔧 Technical Details")
                    st.json({
                        "Predicted Class": "Live" if prediction == 1 else "Spoof",
                        "Confidence Score": f"{confidence:.4f}",
                        "Spoof Probability": f"{probabilities[0]:.4f}",
                        "Live Probability": f"{probabilities[1]:.4f}",
                        "Device": str(device).upper()
                    })
        else:
            st.info("👆 Upload an image or take a photo to get started")
    
    # Sample images for testing
    st.header("🖼️ Sample Test Images")
    st.write("Don't have test images? Try these sample scenarios:")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        st.write("**📱 Phone/Tablet Display**")
        st.write("- Photo of a photo")
        st.write("- Video on screen")
        
    with sample_col2:
        st.write("**👤 Real Person**")
        st.write("- Live selfie")
        st.write("- Direct camera capture")
        
    with sample_col3:
        st.write("**🎭 Mask/Print**")
        st.write("- Paper printouts")
        st.write("- 3D masks")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Streamlit & PyTorch** | Face Anti-Spoofing Detection System")

if __name__ == "__main__":
    main()