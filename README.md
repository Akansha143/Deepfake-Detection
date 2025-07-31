# Deepfake-Detection

Deepfake Detection Project

This repository contains the code and resources for a deepfake detection system developed using ConvNeXt and ViT models. The project focuses on preprocessing video datasets to extract cropped facial regions, enhancing detection accuracy. The models were trained and tested on the **140k Real and Fake Faces** dataset from Kaggle:  
[140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).

---

## Features

- **Dataset Preprocessing:** Videos are processed to focus solely on cropped facial regions for improved model accuracy.  
- **Model Architectures:** Implementation of ConvNeXt and ViT with and without fine-tuning.  
- **Augmentation Techniques:** Utilized AutoAugment and RandAugment for data enhancement.  
- **Streamlit Integration:** A user-friendly interface for real-time deepfake detection.

---

## Dependencies

The following dependencies are required to run the project:

- Python 3.8+  
- PyTorch  
- torchvision  
- Streamlit  
- OpenCV  
- NumPy  
- Matplotlib  
- scikit-learn  
- tqdm

Install all dependencies with:

```bash
pip install -r requirements.txt

## Dataset

This project makes use of the **140k Real and Fake Faces** dataset, available on Kaggle:  
[140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).

Ensure that the dataset is downloaded and placed correctly in the project directory before running the application.

---

## How to Run

Follow these steps to run the deepfake detection Streamlit app:

### Clone the repository:

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection

