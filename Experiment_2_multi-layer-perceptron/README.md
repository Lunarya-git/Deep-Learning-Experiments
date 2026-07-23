# Multi-Layer-Perceptron

Implementation of a Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

## Overview

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)** using **TensorFlow/Keras** for multi-class image classification as part of the **CS3807 – Deep Learning Laboratory (Experiment 2)**.

The experiment demonstrates the complete deep learning workflow, including:

- Dataset exploration
- Image preprocessing
- MLP model construction
- Model training
- Performance evaluation
- Hyperparameter optimization using Randomized Search
- Comparison of baseline and optimized models

---

## Objective

To implement a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for classifying Fashion-MNIST images, evaluate its performance, and improve classification accuracy through automated hyperparameter optimization.

---

## Dataset

**Dataset:** Fashion-MNIST

**Source:** TensorFlow/Keras Datasets

https://github.com/zalandoresearch/fashion-mnist

### Features

- 28 × 28 grayscale clothing images
- Flattened into 784-dimensional feature vectors for the MLP

### Target Classes

- 0 → T-shirt/Top
- 1 → Trouser
- 2 → Pullover
- 3 → Dress
- 4 → Coat
- 5 → Sandal
- 6 → Shirt
- 7 → Sneaker
- 8 → Bag
- 9 → Ankle Boot

Training samples: **60,000**

Testing samples: **10,000**

Number of classes: **10**

Missing values: **None**

---

## Repository Contents

| File | Description |
|------|-------------|
| multi_layer_perceptron.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| plots/ | Generated plots (optional) |
| results/ | Saved outputs (optional) |

---

## Experiments Performed

### Task 1
- Dataset Exploration
- Dataset dimensions
- Sample image visualization
- Class distribution

### Task 2
Data Preprocessing

- Flatten images (28×28 → 784)
- Normalize pixel values
- One-hot encode labels

### Task 3
Model Construction

- Input Layer (784 neurons)
- Dense Layer (128 neurons, ReLU)
- Dense Layer (64 neurons, ReLU)
- Output Layer (10 neurons, Softmax)

### Task 4
Training

- Adam Optimizer
- Categorical Crossentropy Loss
- Accuracy Metric
- Model training with validation

### Task 5
Evaluation

- Test Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Additional Experiments

- RandomizedSearchCV for hyperparameter optimization
- Hyperparameter tuning using SciKeras
- Baseline vs Optimized model comparison
- Training and validation performance analysis

---

## Hyperparameters Optimized

The following hyperparameters were explored:

- Number of Hidden Layers
- Number of Hidden Neurons
- Learning Rate
- Optimizer
- Activation Function
- Batch Size
- Number of Epochs
- Dropout Rate

Optimization Method:

- RandomizedSearchCV
- 5-Fold Cross Validation
- SciKeras Wrapper

---

## Results

The optimized MLP model achieved improved performance over the baseline model.

Performance metrics evaluated include:

- Accuracy
- Precision
- Recall
- F1-score
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

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
multi_layer_perceptron.ipynb
```

Run all cells sequentially.

---

## Author

Aishwarya Muthukumar
