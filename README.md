Capuchin Bird Call Classification
Overview

This project implements a binary audio classification system that distinguishes between capuchin bird calls and non-capuchin sounds using a Convolutional Neural Network (CNN). Audio clips are converted into log-mel spectrograms, which are treated as image inputs for the model.

The pipeline includes dataset downloading, preprocessing, feature extraction, model training, and evaluation.

Code Structure

The code is organized into the following sections:

Setup & Dependencies
Installs required libraries (TensorFlow, librosa, kagglehub).
Imports
Loads all necessary Python packages for audio processing, machine learning, and evaluation.
Dataset Download
Automatically downloads the dataset using kagglehub.
Defines paths for:
Capuchin audio clips
Non-capuchin audio clips
Audio Configuration
Sets sampling rate (16 kHz)
Fixes audio clip duration (3 seconds)
Audio Preprocessing
load_audio: Loads and pads/truncates audio
extract_log_mel: Converts audio to log-mel spectrogram
normalize: Standardizes feature values
Dataset Construction
load_dataset:
Loads all audio files
Converts them into spectrograms
Assigns labels:
1 = capuchin
0 = not capuchin
Data Preparation
Shuffles dataset
Resizes spectrograms to (128, 128)
Splits into:
Training set
Validation set
Test set
Model Definition
CNN with:
3 convolutional layers
Batch normalization
Max pooling
Fully connected layers
Dropout for regularization
Training
Uses:
Early stopping
Learning rate reduction
Evaluation
Outputs:
Confusion matrix
Classification report
Test accuracy
