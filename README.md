# Deep Learning Project

This project involves training a Convolutional Neural Network (CNN) for image classification. The dataset consists of images categorized into 36 classes. The model is trained on a custom dataset using TensorFlow and Keras.

---

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Google Colab (if running in a Colab environment)

---

## Dataset

- Training Data: `/content/drive/MyDrive/Deep learning project/train` (3119 files belonging to 36 classes)
- Validation Data: `/content/drive/MyDrive/Deep learning project/validation` (351 files belonging to 36 classes)

---

## Model Architecture

The CNN model includes:
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dropout layers to reduce overfitting
- Dense (fully connected) layers
- Output layer with softmax activation for multi-class classification

---

## Code

### 1. Mount Google Drive and Load Dataset

```python
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load Training Set
training_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Deep learning project/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True
)

# Load Validation Set
validation_set = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/Deep learning project/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True
)
