import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .big-title {
        font-size:50px !important;
        font-weight:bold !important;
        color:#2A76BE;
        text-align:center;
        padding-bottom:30px;
    }
    .result-text {
        font-size:30px !important;
        font-weight:bold !important;
        text-align:center;
        padding:20px;
    }
    .section-header {
        font-size:35px !important;
        font-weight:bold !important;
        color:#2A76BE;
        padding:10px;
        border-bottom:2px solid #2A76BE;
    }
    .confidence-bar {
        background-color:#2A76BE;
        padding:20px;
        border-radius:10px;
        color:white;
        font-size:24px;
        text-align:center;
    }
</style>
""", unsafe_allow_html=True)

# Load pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('xray_best_model.h5')
    return model

model = load_model()

# Main dashboard
def main():
    st.markdown('<p class="big-title">🩺 Pneumonia Detection from Chest X-rays</p>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<p class="section-header">Upload Chest X-ray Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="file_uploader")
    
    if uploaded_file is not None:
        # Display and process image
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array_expanded, verbose=0)
        probability = prediction[0][0]
        
        # Calculate confidence properly
        confidence = probability if probability > 0.5 else 1 - probability
        confidence_percent = confidence * 100
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="section-header">Original Image</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f'<div class="confidence-bar">Confidence: {confidence_percent:.2f}%</div>', 
                        unsafe_allow_html=True)
            
            # Result text
            if probability > 0.5:
                st.markdown('<p class="result-text" style="color:#FF4B4B;">Pneumonia Detected</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="result-text" style="color:#00C853;">Normal Chest X-ray</p>', 
                           unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()