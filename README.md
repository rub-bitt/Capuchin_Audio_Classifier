# Capuchin Bird Call Classification

## Overview

This project builds a binary audio classifier to distinguish between **capuchin bird calls** and **non-capuchin sounds** using a Convolutional Neural Network (CNN). Audio clips are converted into log-mel spectrograms, which are treated as image inputs for classification.

---

## Code Structure

The code is organized into the following components:

- **Setup & Imports**  
  Installs and loads required libraries (TensorFlow, librosa, kagglehub, etc.).

- **Dataset Download**  
  Automatically downloads the dataset and defines paths for capuchin and non-capuchin audio clips.

- **Audio Preprocessing**
  - `load_audio`: Loads and standardizes audio length  
  - `extract_log_mel`: Converts audio to spectrogram  
  - `normalize`: Scales features for stable training  

- **Dataset Construction**
  - `load_dataset`: Loads audio files, converts to spectrograms, assigns labels  
    - 1 = capuchin  
    - 0 = not capuchin  

- **Data Preparation**
  - Shuffles dataset  
  - Resizes inputs to (128, 128)  
  - Splits into train, validation, and test sets  

- **Model**
  - CNN with convolutional layers, batch normalization, pooling, and dropout  

- **Training**
  - Uses early stopping and learning rate scheduling  

- **Evaluation**
  - Outputs test accuracy  

- **Saving**
  - Saves trained model as `capuchin_classifier.h5`  

---
## Code Attribution

### Written by Me
- `load_dataset` function (data loading and labeling)
- CNN model architecture
- Training and evaluation pipeline
- train/validation/test split

### Adapted from Prior Code
- Audio preprocessing functions:
  - `load_audio`
  - `extract_log_mel`
  - `normalize`
- These were modified to:
  - Ensure fixed-length audio clips
  - Standardize feature scaling
  - Integrate cleanly into the training pipeline

### Modifications Made (with Specific Changes)

- These were modified to:
  - Ensure fixed-length audio clips
  - Standardize feature scaling
  - Integrate cleanly into the training pipeline

## Dependencies

Install required packages:

```bash
pip install tensorflow tensorflow-io librosa kagglehub scikit-learn numpy



