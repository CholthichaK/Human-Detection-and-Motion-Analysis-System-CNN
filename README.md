# Human Detection and Motion Analysis System

A deep learning-based system for detecting and classifying human motions using Python and TensorFlow. The project processes video data, extracts features, and uses a Convolutional Neural Network (CNN) to recognize different human activities.

---

## Features

- Human motion analysis and classification
- Data preprocessing and dataset generation
- Feature extraction pipeline
- CNN-based activity recognition model
- Trained model files included
- Simple graphical user interface (GUI)
- Prediction on new input data

---

## Activities Detected

The system can classify the following actions:

- Boxing
- Hand Clapping
- Hand Waving
- Jogging
- Running
- Walking

---

## Project Structure

```text
Human Detection and Motion Analysis System
│
├── dataset/                     # Original dataset
├── KTHdataset/                  # KTH action dataset
├── processed_data/              # Processed features
├── processed_sequences/         # Sequence data
├── dataset_split.py             # Dataset splitting
├── preprocessing.py             # Data preprocessing
├── feature_extraction.py        # Feature extraction
├── motion_dataset_generator.py  # Dataset generation
├── sequence_dataset_generator.py
├── train.py                     # Model training
├── train_cnn.py                 # CNN training
├── predict.py                   # Activity prediction
├── gui.py                       # User interface
├── scaler.save                  # Saved scaler
├── motion_classifier.keras      # Trained model
└── final_motion_model.h5        # Final model
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## Dataset

This project uses the **KTH Human Action Dataset**, which contains six different human activities:

- Walking
- Jogging
- Running
- Boxing
- Hand Clapping
- Hand Waving

---

## Model

The activity recognition model is built using a Convolutional Neural Network (CNN) trained on extracted motion features to classify human actions.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Human-Detection-and-Motion-Analysis-System.git
```

### Install required packages

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the CNN model

```bash
python train_cnn.py
```

### Run prediction

```bash
python predict.py
```

### Launch the GUI

```bash
python gui.py
```

---

## Results

The CNN model achieved approximately **64% test accuracy** on the KTH Human Action Dataset.

---

## Future Improvements

- Improve model accuracy
- Add real-time webcam detection
- Explore LSTM and Transformer-based models
- Deploy the system as a web application
- Support additional human activities

---

## Author

Developed as a Machine Learning and Computer Vision project.
