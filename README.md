# American Sign Language (ASL) Detection 🖐️

## Objective
The objective of this project is to build a system that can detect a given American Sign Language (ASL) input image and output what the sign represents (such as letters of the alphabet A-Z, space, delete, or nothing).

## Dataset
The dataset contains 29 classes of signs:
*   26 classes for the letters A-Z
*   3 classes for SPACE, DELETE, and NOTHING

## Features
*   **Machine Learning Model:** A deep learning Convolutional Neural Network (CNN) trained using TensorFlow and Keras to classify hand gestures.
*   **Web Interface:** An interactive Streamlit frontend that allows users to upload an image and instantly get the predicted sign and confidence score.

## How to Run Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Files in this Repository
*   `ASL_Detection.ipynb`: The Jupyter Notebook containing data preprocessing, model building, and training.
*   `app.py`: The Streamlit web application.
*   `asl_model.h5`: The saved trained CNN model.
*   `ASL_Detection_Report.pdf`: The detailed project report.
*   `requirements.txt`: Python dependencies.
