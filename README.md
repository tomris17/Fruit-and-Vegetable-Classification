# Fruit & Vegetable Binary Classification with Custom CNN & MobileNetV2

An end-to-end Deep Learning project to automatically classify images into two primary food categories (Fruit vs. Vegetable) using Custom CNN and **MobileNetV2 Transfer Learning**, deployed via an interactive Streamlit web interface.

---

## Project Overview

This repository evaluates custom convolutional representations against pre-trained **MobileNetV2 Transfer Learning** features for high-level binary classification (Fruit vs. Vegetable), maintaining low memory consumption and fast execution.

* **Dataset:** Fruit-Vegetables Dataset
* **Architectures:** Custom CNN & MobileNetV2 Transfer Learning
* **Task Type:** Binary Classification (`Fruit` vs. `Vegetable`)
* **Deployment:** Streamlit Web Application

---

## Performance & Results

| Architecture | Training Accuracy | Validation Accuracy | Validation Loss |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | ~96.20% | **96.70%** | ~0.1210 |
| **MobileNetV2 (Transfer Learning)** | **88.31%** | **95.73%** | **0.1520** |

* **Loss Function:** `binary_crossentropy`
* **Output Activation:** `sigmoid` / `softmax`

---

## Key Features

* **Custom Subfolder Mapping:** Maps multiple product categories directly into high-level binary targets (`Fruit` / `Vegetable`).
* **Dual Architecture Comparison:** Benchmarks a custom multi-layer CNN against pre-trained **MobileNetV2** representations.
* **Regularization:** Uses Dropout (0.3) to prevent overfitting during training.
* **Interactive UI:** Streamlit application for real-time image uploads and immediate predictions with confidence scores.

---

## Tech Stack

* **Python 3.x**
* **TensorFlow / Keras**
* **Streamlit**
* **OpenCV (`cv2`) & Pillow**
* **NumPy, Pandas, Matplotlib, Scikit-Learn**

---

## Dataset Structure

```text
Fruit-Vegetables-Dataset/
├── train/       (Fruit, Vegetable)
├── validation/  (Fruit, Vegetable)
└── test/        (Fruit, Vegetable)
