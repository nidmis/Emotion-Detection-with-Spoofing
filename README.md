**AI Facial Insights Platform ✨**
Version: 1.1.0

**📖 Overview**-
The AI Facial Insights Platform is a sophisticated Streamlit web application designed for comprehensive facial analysis. It leverages deep learning models to perform real-time emotion detection and authenticity (spoof) verification from various sources, including live webcam feeds, uploaded images, and video files. The platform also provides a dashboard to view aggregated analysis statistics.

**🚀 Features**-
Multi-Modal Input:

Live Webcam Analysis: Real-time emotion and spoof detection.

Image Upload Analysis: Analyze static images for facial attributes.

Video File Analysis: Process video files frame-by-frame.

Core Analyses:

Emotion Detection: Identifies emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Spoof Detection (Authenticity Verification): Determines if a face is real or a spoof attempt (e.g., photo, screen).

Interactive Controls:

Adjustable spoof detection threshold for fine-tuning sensitivity.

Frame skipping and maximum frame limits for video analysis.

Visual Feedback:

Annotated video/image display with bounding boxes and predicted labels.

Detailed result cards for each detected face, showing emotion and authenticity with confidence scores.

Emotion probability charts.

Analysis Dashboard:

Metrics on total analyses, unique emotions, overall authenticity rate, and average confidence.

Visualizations of emotion distribution and analysis activity over time.

Table of recent analysis results.

Option to clear analysis history.

**🛠️ Technologies Used**-
The platform is built using Python and leverages several powerful libraries and frameworks:

Web Framework: Streamlit

Deep Learning: TensorFlow (with Keras API)

Computer Vision: OpenCV (cv2)

Numerical Computation: NumPy

Data Handling: Pandas

Image Processing: Pillow (PIL)

Visualization: Plotly (Plotly Express), Matplotlib

Machine Learning Utilities: Scikit-learn

Dataset Acquisition (for training): Kaggle API

For a detailed list of Python packages and their versions, please refer to the requirements.txt file.

**⚙️ Setup and Installation**-
Prerequisites:

Python 3.8 - 3.11

pip (Python package installer)


Install Dependencies:
Ensure you have the requirements.txt file (as provided in the immersive artifact streamlit_requirements_txt) in your project root.

pip install -r requirements.txt

Download Model Files and Haar Cascade:
The following files are required and must be placed in the same directory as app.py (the main Streamlit script):

Emotion Detection Model: fer2013_emotion_model_augmented_best.h5

Spoof Detection Model: spoof_detection_mobilenet_finetuned_best.h5

Face Detection Cascade: haarcascade_frontalface_default.xml (Can be downloaded from the official OpenCV GitHub repository)

**▶️ Running the Application**-
Once the setup is complete, you can run the Streamlit application using the following command in your terminal (from the project's root directory):

streamlit run app.py

This will typically open the application in your default web browser.

**🧠 Model Files**-
fer2013_emotion_model_augmented_best.h5: Pre-trained Keras model for classifying facial emotions.

spoof_detection_mobilenet_finetuned_best.h5: Pre-trained Keras model, likely based on MobileNet, for detecting spoofing attempts.

haarcascade_frontalface_default.xml: OpenCV's pre-trained Haar Cascade classifier for detecting frontal faces in images/video frames.

**🏋️ Model Training**-
The pre-trained models used in this application were developed using separate training processes. The Jupyter notebooks provided (Emotion_Detection_Training (1).ipynb and Spoof_Detection_Training_(Real_Fake_Face)_for_Colab (1).ipynb) outline these training workflows. If you wish to retrain or fine-tune these models, you can refer to these notebooks. They typically involve:

Dataset preparation (e.g., FER2013, custom real/fake face datasets).

Data augmentation.

Model architecture definition (e.g., CNNs, MobileNet).

Model compilation and training.

Evaluation and saving the best model.

Ensure you have the necessary datasets and computational resources (like a GPU, as indicated for Colab usage) if you plan to run the training scripts.

 
