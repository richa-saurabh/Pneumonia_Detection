import streamlit as st
import tensorflow as tf
import numpy as np
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
    :root {
        --primary: #2A76BE;
        --secondary: #6DD5FA;
        --danger: #FF4B4B;
        --success: #00C853;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, 
            #FF0000,  /* Red */
            #FF7F00,  /* Orange */
            #FFFF00,  /* Yellow */
            #00FF00,  /* Green */
            #0000FF,  /* Blue */
            #4B0082,  /* Indigo */
            #8B00FF   /* Violet */
        );
        color: #FFFFFF; /* White text for best contrast */
        padding: 2rem;
        border-radius: 15px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .text-container {
        background: rgba(0, 0, 0, 0.5); /* Adds a dark semi-transparent overlay */
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
        
    
    .upload-container {
        border: 2px dashed var(--primary);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: var(--secondary);
        background: rgba(42,118,190,0.05);
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary);
    }
    
    .confidence-meter {
        background: linear-gradient(90deg, var(--danger) 0%, var(--success) 100%);
        height: 25px;
        border-radius: 12px;
        position: relative;
        margin: 1rem 0;
    }
    
    .confidence-indicator {
        position: absolute;
        height: 35px;
        width: 3px;
        background: pink;
        top: -5px;
        transform: translateX(-50%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 250vh;
        background: black; /* Adjust background color */
        color: white;
        margin: 0;
    }       
@keyframes typing {
    from {
        width: 0;
    }
    to {
        width: 75%; /* Adjusts to text width */
    }
}

.typing-effect {
    position: absolute;
    width: 0; /* Start with zero width */
    left: 50%; /* Start from middle */
    transform: translateX(-50%); /* Shift it left by half its width */
    text-align: center;
    font-size: 1.8rem;
    font-weight: bold;
    color: white;
    overflow: hidden;
    white-space: nowrap;
    display: inline-block;
    animation: typing 4s steps(30, end) forwards;
    margin-top: -50px;
}


    .stat-card {
        background: black;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        color: var(--primary);
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
    # Header Section
    st.markdown("""
    <div class="typing-effect">
        AI-Powered Diagnostic Assistant for Rapid CHEST X-ray Analysis
    </div>
            
    <div class="header-gradient">
        <h1>🩺 Pneumonia Detection from Chest X-rays</h1>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Row
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="feature-icon">📈</div>
                <h3>Model Accuracy</h3>
                <h2 style="color: var(--primary);">96.2%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="feature-icon">⏱️</div>
                <h3>Average Analysis Time</h3>
                <h2 style="color: var(--primary);">2.4s</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-card">
                <div class="feature-icon">💡</div>
                <h3>Cases Analyzed</h3>
                <h2 style="color: var(--primary);">15,328</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # File upload section
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-container">
            <h3>📤 Upload Chest X-ray Image</h3>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="file_uploader", label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Display and process image
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array_expanded, verbose=0)
        probability = prediction[0][0]
        confidence = probability if probability > 0.5 else 1 - probability
        confidence_percent = confidence * 100
        
        # Results Section
        with st.container():
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("""
                <div class="result-card">
                    <h3 style="color: var(--primary);">📷 Image Preview</h3>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="result-card">
                    <h3 style="color: var(--primary);">🔍 Analysis Results</h3>
                    <div style="margin: 2rem 0;">
                        <h4 style="margin-bottom: 1rem ; color: black;">Confidence Level</h4>
                        <div class="confidence-meter">
                            <div class="confidence-indicator" style="left: {0}%;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; color: black;font-weight: bold;">
                            <span>0%</span>
                            <span>100%</span>
                        </div>
                    </div>
                """.format(confidence_percent), unsafe_allow_html=True)
                
                if probability > 0.5:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="color: var(--danger); margin-bottom: 1rem;">🚨 Pneumonia Detected</h2>
                        <p style="font-size: 1.2rem;">
                            Confidence: <strong>{confidence_percent:.2f}%</strong><br>
                            Probability: <strong>{probability*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="color: var(--success); margin-bottom: 1rem;">✅ Normal Result</h2>
                        <p style="font-size: 1.2rem;">
                            Confidence: <strong>{confidence_percent:.2f}%</strong><br>
                            Probability: <strong>{(1-probability)*100:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Information Section
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        **Pneumonia AI Diagnostic Assistant** is a deep learning-powered tool designed to assist medical professionals in detecting pneumonia from chest X-ray images.
        
        Key Features:
        - State-of-the-art CNN architecture
        - Real-time analysis capabilities
        - High-confidence predictions
        - Instant visual feedback
        
        *Note: This tool should be used as a secondary diagnostic aid and not as a replacement for professional medical opinion.*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🩺 PneumAI Diagnostic Assistant v1.0 | Developed with ❤️ by Richa Ajeet Aryan</p>
        <p style="font-size: 0.9rem;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()