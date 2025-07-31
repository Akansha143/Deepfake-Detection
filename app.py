import streamlit as st
import cv2
import tempfile
import os
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import numpy as np

# Define the app title
st.title("Deepfake Detection")

# Load the trained model
@st.cache_resource
def load_model():

    BASE_DIR = "/Users/ananyapurkait/Study Files and Folders/Semester Study/Sem VII/CS435/Project/working/code/"  # Replace with the actual path
    MODEL_PATH = os.path.join(BASE_DIR, "convnext_scenario_1.h5")
    
    # Assuming you have a pre-trained ConvNeXt model structure
    model = models.convnext_tiny(pretrained=False)  # Initialize the ConvNeXt model
    model.load_state_dict(torch.load(MODEL_PATH))  # Load the model weights

    # Modify the classifier for binary classification
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocessing function for video frames
@st.cache_resource
def get_preprocessing():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Process a single frame
def preprocess_frame(frame, preprocess):
    """
    Resize, normalize, and convert frame to tensor for PyTorch model input.
    """
    frame_tensor = preprocess(frame)
    return frame_tensor.unsqueeze(0)  # Add batch dimension

# Process video and detect deepfake
def detect_deepfake(video_path, model, preprocess):
    """
    Analyze a video to detect if it's real or a deepfake using the PyTorch model.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video at {video_path}.")
        return None, None

    frame_count = 0
    fake_count = 0
    real_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames

        frame_count += 1
        if frame_count % 10 == 0:  # Process every 10th frame for efficiency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            processed_frame = preprocess_frame(frame_rgb, preprocess)
            with torch.no_grad():  # No need to track gradients for inference
                outputs = model(processed_frame)
                prediction = torch.sigmoid(outputs).item()  # Apply sigmoid to get probability
            predictions.append(prediction)

            if prediction > 0.5:
                fake_count += 1
            else:
                real_count += 1

    cap.release()

    if len(predictions) == 0:
        st.error("Error: No frames were processed.")
        return None, None

    # Calculate accuracy percentage
    total_frames = fake_count + real_count
    if total_frames == 0:
        st.error("No frames processed.")
        return None, None

    accuracy = (max(fake_count, real_count) / total_frames) * 100
    return ("Fake" if fake_count > real_count else "Real"), accuracy

# Load the model and preprocessing
model = load_model()
preprocess = get_preprocessing()

# File uploader
uploaded_video = st.file_uploader("Upload a video file (mp4 format only)", type=["mp4"])

if uploaded_video is not None:
    # Save the uploaded video temporarily
    temp_dir = tempfile.TemporaryDirectory()
    video_path = os.path.join(temp_dir.name, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(uploaded_video)  # Display the video

    st.write("Processing the video...")
    with st.spinner("Analyzing frames..."):
        # Run the detection
        result, accuracy = detect_deepfake(video_path, model, preprocess)
        temp_dir.cleanup()  # Clean up the temporary directory

    # Display results
    if result is not None:
        if result == "Real":
            st.success(f"The video is classified as **{result}** with an accuracy of **{accuracy:.2f}%**")
        else:
            st.error(f"The video is classified as **{result}** with an accuracy of **{accuracy:.2f}%**")
    else:
        st.error("Video classification failed.")


