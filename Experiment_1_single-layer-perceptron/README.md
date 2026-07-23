# Single-Layer-Perceptron

Implementation of a Single Layer Perceptron for Binary Classification

## Overview

This repository contains the implementation of a **Single Layer Perceptron from scratch** for binary classification as part of the **CS3807 – Deep Learning Laboratory (Experiment 1)**.

The experiment demonstrates the complete machine learning workflow, including:

- Dataset exploration
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Perceptron implementation from scratch
- Model training
- Performance evaluation
- Comparison with Scikit-learn's Perceptron
- Additional experiments on learning rate, activation functions, XOR problem, and feature normalization.

---

## Objective

To understand the working of an artificial neuron and implement a Single Layer Perceptron from scratch using the Perceptron Learning Algorithm for binary classification.

---

## Dataset

**Dataset:** Banknote Authentication Dataset

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/267/banknote+authentication

### Features

- Variance
- Skewness
- Curtosis
- Entropy

### Target

- 0 → Authentic Banknote
- 1 → Forged Banknote

Number of samples: **1372**

Number of features: **4**

Missing values: **None**

---

## Repository Contents

| File | Description |
|------|-------------|
| single_layer_perceptron.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| dataset/ | Dataset used for training |
| images/ | Generated plots (optional) |

---

## Experiments Performed

### Task 1
- Dataset Exploration

### Task 2
- Histograms
- Correlation Heatmap
- Scatter Plots
- Boxplots

### Task 3
- Min-Max Feature Normalization
- Train-Test Split (80:20)

### Task 4
Perceptron implemented from scratch:

- Weight Initialization
- Bias Initialization
- Step Activation Function
- Forward Pass
- Perceptron Learning Rule

### Task 5
Training

- Epoch-wise updates
- Weight evolution
- Bias evolution
- Misclassified samples

### Task 6
Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Additional Experiments

- Step vs Sigmoid Activation
- Comparison with Scikit-learn Perceptron
- Learning rate comparison
- XOR problem
- Effect of Feature Normalization

---

## Results

Test Accuracy: **98.91%**

Precision: **100%**

Recall: **97.64%**

F1-score: **98.80%**

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
single_layer_perceptron.ipynb
```

Run all cells sequentially.

---

## Author

Aishwarya Muthukumar

B.Tech Artificial Intelligence & Data Science

Shiv Nadar University Chennai
