# CNN-Transfer-Learning-Architecture-Comparison
Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

## Overview
This repository contains the implementation of **transfer learning with pretrained Convolutional
Neural Networks** using **TensorFlow/Keras**, along with a **from-scratch comparison of five
landmark CNN architectures**, as part of the **CS3807 – Deep Learning Laboratory (Experiment 4)**.

The experiment demonstrates the complete transfer-learning workflow, including:
- CIFAR-10 dataset preparation
- Transfer learning with a frozen pretrained VGG16 base
- Baseline training and fine-tuning (unfreezing the last convolutional block)
- Full performance evaluation
- A six-way hyperparameter study
- A five-architecture comparison (LeNet-5, AlexNet, VGG16, GoogLeNet, ResNet50)

---

## Objective
To understand and implement transfer learning using pretrained CNN models, fine-tune a pretrained
network on CIFAR-10, evaluate its classification performance, and compare it against classical and
modern CNN architectures trained from scratch.

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
| Experiment_4_CNN_Transfer_Learning.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| plots/ | Generated plots (600 DPI) |
| Experiment_4_Report.pdf | Full lab report |

---

## Experiments Performed

### Task 1
Dataset Preparation
- Load CIFAR-10 using TensorFlow/Keras
- Normalize pixel values to [0, 1]
- Display ten sample images
- Print training/testing dataset dimensions

### Task 2
Transfer Learning Model
- Pretrained VGG16 base (ImageNet weights)
- Convolutional base frozen
- Global Average Pooling layer added
- Dense layer with ReLU activation added
- Softmax output layer added

### Task 3
Baseline Training
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 15 (within manual's 10–20 range)
- Loss: Categorical Cross-Entropy
- Metric: Accuracy

### Task 4
Fine-Tuning
- Last convolutional block (block5) unfrozen
- Trained for an additional 8 epochs (within manual's 5–10 range)
- Accuracy compared before vs. after fine-tuning

### Task 5
Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Hyperparameter Study
One-factor-at-a-time comparison on a stratified data subset, covering every value specified by the
manual:
- Learning Rate: 0.001, 0.0001
- Batch Size: 16, 32, 64
- Epochs: 10, 20
- Optimizer: Adam, SGD
- Dense Units: 128, 256
- Frozen Layers: All, Partial

### Architecture Comparison
LeNet-5, AlexNet, VGG16, GoogLeNet, and ResNet50 — implemented via their canonical implementations
(`torchvision.models` for AlexNet/GoogLeNet/VGG16/ResNet50, a faithful from-scratch build for
LeNet-5), trained from scratch under a common, documented training budget, and compared on:
- Total Parameters
- Test Accuracy
- Training Time

### Discussion Questions
All 10 discussion questions from the manual answered (AlexNet's breakthrough, VGG16's 3×3 filter
choice, Inception module advantages, residual learning, LeNet vs. ResNet, transfer learning,
fine-tuning, dilated vs. transpose convolution, pretrained model convergence, and computational
complexity comparison).

### Additional Exercises
1. Transfer learning implemented using VGG16
2. Full experiment repeated using ResNet50
3. Adam vs. SGD optimizer comparison
4. Frozen-layer vs. fine-tuned training comparison
5. Precision, Recall, and F1-score evaluation
6. LeNet, AlexNet, and ResNet performance comparison

---

## Results
The VGG16 transfer-learning model was trained on CIFAR-10 for 15 baseline epochs plus 8 fine-tuning
epochs, and evaluated on the held-out test set. Performance metrics evaluated include:
- Training / Validation Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

The from-scratch architecture comparison (LeNet-5, AlexNet, VGG16, GoogLeNet, ResNet50) reports
Parameters, Test Accuracy, and Training Time for each model under a common training budget.

*(Update this section with your final results after training, if desired.)*

---

## Dependencies
See `requirements.txt`.

Note: Sections 1–8 and 11 use **TensorFlow/Keras**. The Architecture Comparison section uses
**PyTorch/torchvision** so that AlexNet and GoogLeNet are built from their true canonical
implementations rather than hand-approximated substitutes.

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
Experiment_4_CNN_Transfer_Learning.ipynb
```

Run all cells sequentially. A GPU runtime (e.g. Colab T4 or Kaggle P100/T4) is strongly recommended.

---

## Author
Aishwarya Muthukumar
