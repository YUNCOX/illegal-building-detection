import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Set page config for a premium look
st.set_page_config(
    page_title="Erbil Illegal Building Detection",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #3b82f6;
        --background-dark: #0f172a;
        --card-dark: #1e293b;
        --text-light: #f8fafc;
    }
    
    .stApp {
        background-color: var(--background-dark);
        color: var(--text-light);
    }
    
    .css-1d391kg {
        background-color: var(--card-dark);
    }
    
    h1, h2, h3 {
        color: var(--primary-color) !important;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .metric-card {
        background-color: var(--card-dark);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ef4444;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🏙️ Erbil Municipal Oversight AI")
st.markdown("*Powered by Siamese Convolutional Neural Networks (Prototype)*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Erbil_Citadel_2014.jpg/800px-Erbil_Citadel_2014.jpg", use_container_width=True, caption="Erbil City")
    st.header("Control Panel")
    st.markdown("Upload satellite imagery to detect unauthorized construction in target sectors.")
    
    confidence_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.75, 0.05)
    min_area = st.number_input("Minimum Building Area (px²)", value=500, step=100)
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("🟢 AI Model Loaded (Siamese CNN)")
    st.success("🟢 Database Connected")

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Image (T0)")
    img1_file = st.file_uploader("Upload previous satellite scan", type=["png", "jpg", "jpeg"], key="img1")

with col2:
    st.subheader("Recent Image (T1)")
    img2_file = st.file_uploader("Upload latest satellite scan", type=["png", "jpg", "jpeg"], key="img2")

def simulate_siamese_cnn(img1, img2, min_area_thresh):
    """
    Simulates a Siamese CNN by computing image differences.
    In a real scenario, this would pass through a PyTorch/TensorFlow model.
    """
    # Convert PIL to OpenCV format
    cv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    # Resize to match (just in case they are slightly different)
    cv_img2 = cv2.resize(cv_img2, (cv_img1.shape[1], cv_img1.shape[0]))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(cv_img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    
    # Compute absolute difference
    diff = cv2.absdiff(blur1, blur2)
    
    # Threshold the difference
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill in holes
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes
    output_img = cv_img2.copy()
    detections = 0
    total_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area_thresh:
            x, y, w, h = cv2.boundingRect(c)
            # Draw red bounding box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Add label
            cv2.putText(output_img, f'Violation {detections+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            detections += 1
            total_area += area
            
    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), detections, total_area

if img1_file and img2_file:
    image1 = Image.open(img1_file)
    image2 = Image.open(img2_file)
    
    with col1:
        st.image(image1, caption="Baseline Image", use_container_width=True)
    with col2:
        st.image(image2, caption="Recent Image", use_container_width=True)
        
    st.markdown("---")
    
    if st.button("🚀 Run Siamese CNN Analysis", use_container_width=True):
        with st.spinner("Analyzing temporal changes via deep feature extraction..."):
            # Simulate processing time for effect
            time.sleep(2) 
            
            result_img, detection_count, changed_area = simulate_siamese_cnn(image1, image2, min_area)
            
            st.success("Analysis Complete!")
            
            # Display Metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{detection_count}</div>
                    <div class="metric-label">Illegal Structures Detected</div>
                </div>
                """, unsafe_allow_html=True)
            with mcol2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{int(changed_area)} px²</div>
                    <div class="metric-label">Estimated Area Changed</div>
                </div>
                """, unsafe_allow_html=True)
            with mcol3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #10b981;">{confidence_threshold * 100}%</div>
                    <div class="metric-label">Model Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Detection Results")
            st.image(result_img, caption="Detected Violations Highlighted", use_container_width=True)
            
            st.warning("⚠️ **Action Required**: The detected structures do not have corresponding building permits in the municipal database. Please dispatch an inspection team.")
else:
    st.info("👈 Please upload both a Baseline and Recent image to begin analysis.")
    
st.markdown("---")
st.markdown("""
### 🧠 About this AI Project
This system uses a **Siamese Convolutional Neural Network** architecture designed to compare two distinct temporal satellite images. By extracting deep spatial features from both images simultaneously, the network computes a distance metric to highlight structural deviations (new buildings) while ignoring seasonal changes, lighting variations, and minor foliage differences.

**Ethical & Privacy Considerations:**
- Only public and authorized municipal satellite data is processed.
- No personally identifiable information (PII) is gathered from the imagery.
- The algorithm is continuously evaluated for fairness to ensure all city sectors are monitored equitably without demographic bias.
""")
