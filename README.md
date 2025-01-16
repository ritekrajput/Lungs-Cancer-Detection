# Lung Image Classification Using Transfer Learning

This project implements a deep learning model for classifying lung images into different categories using transfer learning with the **InceptionV3** model. The code demonstrates data preparation, model training, and evaluation steps.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup and Dependencies](#setup-and-dependencies)
5. [Results and Metrics](#results-and-metrics)
6. [Acknowledgments](#acknowledgments)

---

## Project Description

This project classifies lung images into three categories by leveraging pre-trained weights of the InceptionV3 model. The training pipeline involves:

- **Data Augmentation and Preprocessing**: Reading, resizing, and normalizing images.
- **Transfer Learning**: Using InceptionV3 as the base model with frozen layers.
- **Model Fine-tuning**: Adding custom dense layers for classification.
- **Evaluation**: Using confusion matrix and classification reports.

---

## Dataset

https://www.kaggle.com/datasets/subho117/lung-cancer-detection-using-transfer-learning

## Model Architecture

The model is built using **InceptionV3** with the following modifications:

1. Pre-trained layers from InceptionV3 are frozen.
2. Custom dense layers are added:
   - Dense layer with 256 neurons and ReLU activation.
   - Dense layer with 128 neurons and ReLU activation.
   - Output layer with 3 neurons (softmax activation for multi-class classification).

---

## Setup and Dependencies

### Prerequisites

Ensure you have the following libraries installed:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras
- OpenCV
- PIL (Pillow)
- scikit-learn

## Results and Metrics

1. **Validation Accuracy**: Training stops when validation accuracy reaches above 90% (as defined by the custom callback).
2. **Confusion Matrix**:
   A confusion matrix is generated to visualize the performance of the model.
3. **Classification Report**:
   A detailed report is printed showing precision, recall, F1-score, and support for each class.

---

## Acknowledgments

- **TensorFlow/Keras Documentation**: For guidance on transfer learning and callbacks.
- **InceptionV3**: For providing a robust pre-trained model for transfer learning.
- **OpenCV and PIL**: For handling and preprocessing image data.

---

