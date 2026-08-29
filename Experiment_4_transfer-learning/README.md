# Deep-CNN-Architecture-Comparison-Transfer-Learning
Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning
 
## Overview
This repository contains the implementation of a **comparative study of deep CNN
architectures** using **TensorFlow/Keras** as part of the **CS3807 – Deep Learning
Laboratory (Experiment 4)**.
 
The experiment demonstrates the complete transfer learning workflow, including:
- Dataset preparation
- Transfer learning using pretrained VGG16 and ResNet50 (real ImageNet weights)
- Model training and fine-tuning
- Model evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
- Hyperparameter study (learning rate, batch size, optimizer, dense units, frozen layers)
- LeNet-5 and AlexNet built and trained from scratch
- GoogleNet (InceptionV3) as the pretrained Inception representative
- Full 5-way architecture comparison
---
 
## Objective
To study the evolution of deep CNN architectures, implement and fine-tune transfer
learning using pretrained CNN models, and compare the classification performance of
LeNet-5, AlexNet, VGG16, GoogleNet (InceptionV3) and ResNet50 on CIFAR-10.
 
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
| Experiment_4_Deep_Learning_Lab.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| outputs/ | Generated plots and result CSVs (600 DPI) |
| Experiment_4_Report.pdf | Full lab report |
 
---
 
## Experiments Performed
 
### Task 1
Dataset Preparation
- Load CIFAR-10 and normalize pixel values to [0,1]
- Ten sample images
- Training/testing dataset dimensions
### Task 2
Transfer Learning — Model Setup (VGG16)
- Load pretrained ImageNet weights, remove original classifier
- Freeze convolutional base
- Add Global Average Pooling, Dense (ReLU), Softmax output layers
### Task 3
Model Training (VGG16)
- Optimizer: Adam, Learning Rate: 0.001, Batch Size: 32, Epochs: 12
- Loss: Categorical Cross-Entropy
### Task 4
Fine-Tuning (VGG16)
- Unfreeze last convolution block, train 8 more epochs
- Validation accuracy before vs after fine-tuning compared
### Task 5
Model Evaluation (VGG16, fine-tuned)
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Misclassified images
### Section 16: Hyperparameter Study
One-factor-at-a-time comparison across:
- Learning Rate: 0.001, 0.0001
- Batch Size: 16, 32, 64
- Optimizer: Adam, SGD
- Dense Units: 128, 256
- Frozen Layers: All, Partial
### Additional Exercise 2
Repeat the Experiment Using ResNet50
- Same Task 2–5 pipeline, real pretrained ResNet50 ImageNet weights
- Compared against VGG16 on accuracy and training time
### Additional Exercise 6
Compare LeNet-5, AlexNet and ResNet
- LeNet-5 and an adapted AlexNet built and trained from scratch on CIFAR-10
- GoogleNet (InceptionV3, pretrained) used as the Inception representative
- Full 5-model comparison table (Parameters, Accuracy, Training Time)
---
 
## Results
 
The primary model (VGG16, fine-tuned) was trained on CIFAR-10 and evaluated on the
held-out test set:
 
| Metric | Value |
|--------|-------|
| Training Accuracy | 99.55% |
| Testing Accuracy | 85.84% |
| Precision | 85.96% |
| Recall | 85.84% |
| F1-score | 85.84% |
| Training Time | 1825.3 s |
| Total Parameters | 14,848,586 |
 
**Architecture comparison:**
 
| Model | Parameters | Accuracy (%) | Training Time (s) |
|-------|-----------:|--------------:|-------------------:|
| LeNet-5 | 83,126 | 51.21 | 97.2 |
| AlexNet (adapted) | 8,576,906 | 10.00 | 396.8 |
| VGG16 | 14,848,586 | 85.84 | 1825.3 |
| GoogleNet (InceptionV3) | 22,329,898 | 67.04 | 259.8 |
| ResNet50 | 24,114,826 | 85.31 | 414.1 |
 
> **Note:** AlexNet (adapted) failed to converge from scratch on CIFAR-10 — its loss got
> stuck at ln(10) from epoch 1 onward, a dead-network result rather than a real accuracy
> score. This is reported as-is in the full lab report.
 
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
Experiment_4_Deep_Learning_Lab.ipynb
```
 
Run all cells sequentially. In Colab, set `Runtime → Change runtime type → T4 GPU`
before running for reasonable training times.
 
---
 
## Author
Aishwarya Muthukumar
 
