import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image

# Load model
MODEL_PATH = "D:/Pneumonia_Project/best_model_c.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure image has 3 channels
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Custom CSS with Pattern Background
st.markdown(
    """
    <style>
    body {
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        background-color: #E6F7FF;
    }
    .title {
        font-size: 80px;
        font-weight: bold;
        color: #003366;
        text-align: center;
        padding: 15px;
    }
    .subtitle {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #005580;
    }
    .upload-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .result-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown('<h1 class="title">🩺 Pneumonia Disease Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔬 AI-powered chest X-ray analysis</p>', unsafe_allow_html=True)

st.markdown("### 📤 Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="📸 Uploaded Chest X-ray", use_column_width=True)
    
    st.markdown('<div class="upload-box">🔄 Click "Analyze Image" to start the diagnosis</div>', unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Image"):
        with st.spinner("⏳ Processing... Please wait"):
            time.sleep(1.5)
            
            # Preprocess and predict
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0][0]

            confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
            diagnosis = "🦠 Pneumonia Detected" if prediction > 0.5 else "✅ Normal Lungs"

            # Display result
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:#003366; text-align:center;'>🩺 Diagnosis: {diagnosis}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#005580; text-align:center;'>🔬 Confidence Score: {confidence}%</h3>", unsafe_allow_html=True)

            # Alerts
            if prediction > 0.5:
                st.error("⚠️ Pneumonia detected! Please consult a doctor.")
            else:
                st.success("✅ Lungs appear normal.")

            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### ℹ️ About Pneumonia Detection")
st.info(
    "Pneumonia is a lung infection that can be life-threatening if untreated. AI-powered models help in early detection, "
    "allowing for timely medical intervention. This tool provides a preliminary analysis but should not replace a doctor's diagnosis."
)
