# Fruit & Vegetable Binary Classification with CNN & Streamlit Web App

This repository contains a Convolutional Neural Network (CNN) built using TensorFlow/Keras to classify images into two primary food categories: Fruits and Vegetables. The project includes an interactive web application powered by Streamlit for real-time inference.

---

## Project Overview

The main objective of this project is to map multiple individual product sub-categories (e.g., apples, bananas, tomatoes, carrots) into binary high-level targets (Fruit vs. Vegetable) and perform binary image classification with optimized data preprocessing.

---

## Key Features

* **Custom Label Mapping:** Dynamically maps subfolder classes to binary categories (Fruit / Vegetable) without altering the disk structure.
* **Preprocessed Pipeline:** Standardizes input resolution (128x128x3), RGB color channels, and pixel normalization (0-1 range).
* **Convolutional Neural Network:** Built with multiple Conv2D and MaxPooling layers, featuring Dropout regularization to prevent overfitting.
* **Streamlit Web Application:** Simple web interface allowing users to upload custom images and obtain immediate prediction labels along with confidence scores.

---

## Technical Stack & Dependencies

* Python 3.x
* TensorFlow / Keras
* OpenCV (`cv2`)
* Streamlit
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-Learn

---

## Performance & Evaluation

* **Task Type:** Binary Classification (Sigmoid output)
* **Classes:** Fruit (`1`), Vegetable (`0`)
* **Evaluation Metrics:** Accuracy curves, Loss curves, and Confusion Matrix visual analysis.

---

## Streamlit Application Setup

1. Train the model using the provided notebook and save it to the project root directory as `fruit_vegetable_model.keras`.
2. Install the necessary dependencies:
   ```bash
   pip install tensorflow streamlit opencv-python pandas numpy matplotlib seaborn scikit-learn pillow
