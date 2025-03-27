# 🫁 Pneumonia Detection using Deep Learning

## 🔍 Overview
This project aims to detect pneumonia from chest X-ray images using a deep learning model. A user-friendly dashboard has been built that allows users to upload images in formats such as **PNG, JPG, and JPEG**. The model then predicts whether the input image shows signs of **Pneumonia** or **Normal** lungs, providing a confidence score for the prediction.

## 🌟 Features
✅ Upload chest X-ray images in PNG, JPG, or JPEG format.
✅ Model classifies the image as either **Pneumonia** or **Normal**.
✅ Displays confidence and probability score of the prediction.
✅ Simple and interactive dashboard for user-friendly experience.

## 🛠 Tech Stack
- **🖥 Frontend:** Streamlit (for dashboard)
- **⚙️ Backend:** Python, TensorFlow/Keras (for deep learning model)
- **📦 Libraries Used:** OpenCV, NumPy, Matplotlib, PIL

## 🚀 Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/pneumonia-detection.git
   cd pneumonia-detection
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
1. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```
2. **Upload a chest X-ray image.**
3. **View the model’s prediction and confidence score.**

## 📂 Dataset
The model is trained using the **Chest X-ray dataset** from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## 🧠 Model Details
- The deep learning model is based on **CNN (Convolutional Neural Networks)**.
- Pre-trained models like **ResNet50/VGG16** were used for transfer learning.
- Trained on labeled pneumonia and normal lung images.

## 📊 Results
- ✅ Achieved an accuracy of **92%** on the test dataset.
- ✅ Confidence scores provide reliability in predictions.

## 🔮 Future Improvements
- 📈 Improve accuracy by training on a larger dataset.
- 🎨 Enhance the dashboard UI for better user experience.
- 🌐 Deploy the model as a web application.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.



