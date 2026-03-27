# ASL (American Sign Language) Detection Web App 🖐️

**🌐 Live Demo:** [https://asl-detection-web.streamlit.app/](https://asl-detection-web.streamlit.app/)

An interactive Streamlit web application that uses Convolutional Neural Networks (CNNs) to automatically detect American Sign Language (ASL) hand signs from uploaded images. Built as part of the Unified Mentor Internship Project.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

## 🎯 Objective
Build a system that can detect a given ASL input image and output what the sign represents. The model supports 29 classes: the 26 letters of the English alphabet (A–Z) plus 3 special actions (`del`, `space`, `nothing`). Automating sign recognition helps bridge communication gaps.

## ✨ Features
- **Sign Recognition:** Upload a 64x64 or larger image of a hand sign to get the predicted letter or action.
- **Confidence Distribution:** Interactive Plotly chart displaying the model's confidence for the top 5 predicted classes.
- **High Accuracy:** Achieves ~98% accuracy on the test dataset.
- **ASL Reference Guide:** Built-in section reviewing the 29 supported classes for quick reference.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/abhiguru25/ASL-Detection.git
cd ASL-Detection/ASL_detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧠 Methodology & Model
- **Data Preprocessing:** Images are resized to `64 × 64` and pixel values are normalized to `[0, 1]`.
- **Architecture:** Custom CNN optimized for spatial pattern recognition (hand shapes, angles, finger positions).
- **Evaluation:** Evaluated using categorical cross-entropy and Adam optimizer.

## 📁 Repository Structure
```
├── app.py            # Main Streamlit application file
├── requirements.txt  # Python dependencies
├── asl_model.h5      # Pre-trained Keras CNN model (ensure this is present)
└── README.md         # This documentation
```

---
*Developed by Abhivirani | Unified Mentor Internship Project*
