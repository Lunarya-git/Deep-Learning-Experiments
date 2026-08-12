# Convolutional-Neural-Network-Image-Classification
Implementation of a Convolutional Neural Network (CNN) for Image Classification

## Overview
This repository contains the implementation of a **Convolutional Neural Network (CNN)** using
**TensorFlow/Keras** for image classification as part of the **CS3807 – Deep Learning Laboratory
(Experiment 3)**.

The experiment demonstrates the complete CNN workflow, including:
- Dataset exploration
- Convolution operation and kernel size comparison
- Stride and padding study
- Feature map visualization
- Max pooling vs average pooling comparison
- CNN model construction and training
- Performance evaluation

---

## Objective
To understand the working principle of Convolutional Neural Networks by implementing convolution,
pooling, and feature map visualization, and to build a CNN using TensorFlow/Keras for classifying
CIFAR-10 images.

---

## Dataset
**Dataset:** CIFAR-10
**Source:** TensorFlow/Keras Datasets
https://www.cs.toronto.edu/~kriz/cifar.html

### Features
- 32 × 32 RGB colour images
- 3 input channels (Red, Green, Blue)

### Target Classes
- 0 → Airplane
- 1 → Automobile
- 2 → Bird
- 3 → Cat
- 4 → Deer
- 5 → Dog
- 6 → Frog
- 7 → Horse
- 8 → Ship
- 9 → Truck

Training samples: **50,000**
Testing samples: **10,000**
Number of classes: **10**
Missing values: **None**

---

## Repository Contents
| File | Description |
|------|-------------|
| cnn_image_classification.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| plots/ | Generated plots (600 DPI) |
| Experiment_3_Report.pdf | Full lab report |

---

## Experiments Performed

### Task 1
Dataset Exploration
- Ten sample images
- Dataset dimensions
- Class distribution

### Task 2
Convolution Layer
- Kernel size comparison: 3×3, 5×5, 7×7
- Feature map size recorded for each

### Task 3
Hyperparameter Study
- Stride: 1 vs 2
- Padding: Same vs Valid
- Output dimension computation

### Task 4
Feature Map Visualization
- Eight+ feature maps from the first convolution layer

### Task 5
Pooling Comparison
- Max Pooling vs Average Pooling (output size)

### Task 6
Model Construction and Training
- Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → Softmax
- Optimizer: Adam
- Epochs: 20
- Batch Size: 32

### Task 7
Evaluation
- Test Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Additional Exercises
1. Output size calculation for a 64×64 image, 5×5 kernel, stride 2, padding 2
2. Trainable parameter calculation for a 64-filter, 3×3, RGB convolution layer
3. ReLU vs Sigmoid activation comparison
4. Max Pooling vs Average Pooling — full accuracy comparison
5. Effect of increasing filters from 16 to 64 on accuracy and training time

---

## Results
The CNN model was trained on CIFAR-10 for 20 epochs and evaluated on the held-out test set.
Performance metrics evaluated include:
- Training / Validation Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

*(Update this section with your final results after training, if desired.)*

---

## Dependencies
See `requirements.txt`.

---

## Execution Instructions

### 1. Clone the repository
```bash
git clone <your-github-repository-link>
cd <repository-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook (or open in Google Colab)
```bash
jupyter notebook
```
Open:
```
cnn_image_classification.ipynb
```
Run all cells sequentially.

---

## Author
Aishwarya Muthukumar
