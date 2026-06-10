# MoodSync: Real-Time Emotion Recognition & Adaptive Music AI

MoodSync is a real-time facial emotion recognition system built using Convolutional Neural Networks (CNNs). It dynamically detects a user's mood via a live webcam feed and features an adaptive music recommendation system based on the detected emotion.

Developed by Salman Farsi and Rifaat Nuha.

## Features
- **Real-Time Emotion Recognition**: Utilizes a custom VGG-style CNN ("FERNET") to classify facial expressions into 5 categories: Angry, Happy, Neutral, Sad, and Surprise.
- **Robust Face Detection**: Employs OpenCV's DNN module with a ResNet-10 SSD face detector for accurate localization.
- **Music Recommendation**: Automatically suggests music genres (e.g., Pop/EDM for Happy, Ambient Chill for Neutral) based on your current emotional state.
- **Temporal Smoothing**: Uses a sliding buffer to average frame predictions, providing stable, flicker-free output.

## Architecture & Models
Our proposed model, **FERNET**, achieved a **71.45%** accuracy and an AUC of **0.922** on a custom-balanced FER-2013 dataset (10,000 images). We also provide two baseline models for comparison:
1. Custom Lightweight CNN
2. DenseNet121 (Transfer Learning)



## Details & Demonstration

Here is a demonstration of MoodSync in action:

*(Here is the MoodSync demonstration video!)*

https://github.com/imFARSI/MoodSync-Real-Time-Emotion-Recognition-and-Adaptive-Music-AI/raw/master/moodync%20video.mp4

## Setup & Usage
1. Clone this repository.
2. Install the required dependencies (TensorFlow, Keras, OpenCV, NumPy).
3. Run the cells in `MoodSync.ipynb` to explore the data preprocessing, model architectures, and training process.
4. Execute the Real-Time Inference Pipeline cell at the bottom of the notebook to launch the webcam module!
