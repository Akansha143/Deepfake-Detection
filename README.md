# Deepfake Detection Project

This repository contains the code and resources for a deepfake detection system developed using ConvNeXt and ViT models. The project involves preprocessing video datasets to focus on cropped facial regions, thereby improving model accuracy. The models were trained and tested on the **140k Real and Fake Faces** dataset, sourced from Kaggle: [140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/).

## Features

- **Dataset Preprocessing:** Cropped facial regions extracted from videos for better accuracy.  
- **Model Architectures:** Implementation of ConvNeXt and ViT with and without fine-tuning.  
- **Augmentation Techniques:** AutoAugment and RandAugment transformations for data enhancement.  
- **Streamlit Integration:** A user-friendly interface for real-time deepfake detection.

## Dependencies

The project requires the following libraries and frameworks:

- Python 3.7+  
- PyTorch  
- torchvision  
- Streamlit  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-learn  
- tqdm  
- face_recognition  
- tensorflow  

Install dependencies using pip:

```bash
pip install -r requirements.txt
```
## Dataset

The **140k Real and Fake Faces** dataset can be downloaded from Kaggle: [140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/).  
Ensure that the dataset is downloaded and appropriately placed in the project directory.

## How to Run

To launch the Streamlit app for real-time deepfake detection:

**1.Clone the repository:**

```bash
git clone git@github.com:24PAnanya/Deepfake-detection-CS435.git
cd deepfake-detection
```

**2.Run the Streamlit app:**

```bash

streamlit run app.py
```
3.Follow the instructions on the Streamlit interface to upload videos and view detection results.

##Acknowledgments
This project was inspired by the growing need for robust methods to combat deepfakes. The dataset used, 140k Real and Fake Faces, was obtained from Kaggle. Special thanks to the authors of the dataset for their contributions.
