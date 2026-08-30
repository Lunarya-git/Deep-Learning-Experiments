# CNN-Training-Regularization-Optimization-Transfer-Learning-CV
Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

## Overview
This repository contains the implementation of a **comprehensive study of CNN
training and model-selection techniques** using **TensorFlow/Keras** as part of the
**CS3807 – Deep Learning Laboratory (Experiment 5)**.

The experiment demonstrates the complete pipeline for building a reliable image
classifier, including:
- Dataset preparation (Oxford-IIIT Pet, downloaded directly from the official source)
- Weight initialization study (Zero, Random, Xavier/Glorot, He) trained from scratch
- Regularization and overfitting analysis (None, L2, Dropout, Batch Normalization)
- Batch Normalization — numerical example and empirical with-vs-without comparison
- Optimizer comparison (SGD, Momentum, RMSProp, Adam) at a fixed learning rate
- CNN hyperparameter tuning (learning rate, batch size, dropout rate)
- Transfer learning and fine-tuning using pretrained MobileNetV2
- 5-fold stratified cross-validation for model selection
- Final model retraining, test-set evaluation and confusion-matrix analysis
- Additional exercise comparing two new hyperparameter configurations
---

## Objective
To systematically study the effect of weight initialization, regularization,
optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning and
cross-validation on image classification performance, using a single CNN
architecture (MobileNetV2) on the Oxford-IIIT Pet dataset, and to select a final
model configuration using 5-fold cross-validation.

---

## Dataset
**Dataset:** Oxford-IIIT Pet Dataset
**Source:** Official VGG mirror (downloaded directly in the notebook)
https://www.robots.ox.ac.uk/~vgg/data/pets/

### Features
- RGB images of cats and dogs, variable original size
- Resized to 224 × 224 × 3 and normalized with MobileNetV2's `preprocess_input`
### Target Classes
- 37 pet breeds (cats and dogs)

Pool (train+val) samples: **3,680**
Held-out test samples: **3,669**
Train / Validation split (from pool): **3,128 / 552** (stratified, 15% validation)
Number of classes: **37**
Missing values: **None** (a handful of slightly truncated image files are handled with `ImageFile.LOAD_TRUNCATED_IMAGES`)

---

## Repository Contents

| File | Description |
|------|-------------|
| Experiment_5_Deep_Learning_Lab.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| outputs/ | Generated plots and result CSVs (600 DPI) |
| Experiment_5_Report.pdf | Full lab report |

---

## Experiments Performed

### Section 5
Weight Initialization
- MobileNetV2 trained **from scratch** (`weights=None`) with Zero, Random, Xavier
  and He initializers, 6 epochs each
- Training loss and validation accuracy compared across all four strategies
### Section 6
Regularization and Overfitting
- ImageNet-pretrained MobileNetV2 with the base **frozen**
- Classifier head varied: No regularization, L2, Dropout, Batch Normalization
- Training vs. validation accuracy/loss compared to identify the generalization gap
### Section 7
Batch Normalization
- Manual numerical example (mini-batch mean, variance, normalization) verified in code
- Empirical with-BN vs. without-BN validation accuracy comparison
### Section 8
Optimization Algorithms
- SGD, Momentum, RMSProp and Adam compared at the same fixed learning rate (0.001)
- Final loss, best validation accuracy, epochs to converge and training time recorded
### Section 9
CNN Hyperparameter Tuning
- One-factor-at-a-time sweep: Learning Rate (0.001, 0.0001), Batch Size (16, 32, 64),
  Dropout Rate (0, 0.25, 0.5)
- Convolution output-size formula verified numerically
### Section 10
Transfer Learning and Fine-Tuning
- Case A: Feature extraction with a fully frozen pretrained base
- Case B: Fine-tuning — 3-epoch frozen warm-up, then top layers unfrozen at a
  smaller learning rate
- Validation accuracy and loss compared before/after unfreezing
### Section 11
5-Fold Cross-Validation
- Four candidate configurations (baseline, best regularization, best tuned
  hyperparameters, fine-tuned) evaluated with stratified 5-fold CV on the training pool
- Mean and standard deviation of fold accuracies compared
### Section 12
Final Model Evaluation
- CV-selected configuration retrained on the complete training pool
- Evaluated once on the untouched test set: accuracy, precision, recall, F1-score
- Confusion matrix and sample misclassified images analyzed
### Section 16: Additional Exercise
Two New Hyperparameter Configurations
- Two new learning-rate/dropout/batch-size/optimizer combinations evaluated with
  5-fold cross-validation
- Compared against the Section 11–12 winning configuration on mean accuracy, SD
  and computational cost
---

## Results

The final selected configuration (fine-tuned MobileNetV2, Dropout head, Adam
optimizer, lr 0.001 → 1e-5 after warm-up, batch size 16, dropout 0.25) was
retrained on the full training pool and evaluated on the held-out test set:

| Metric | Value |
|--------|-------|
| Mean CV Accuracy | 91.25% |
| CV Standard Deviation | 1.20 |
| Test Accuracy | 88.96% |
| Precision (macro) | 89.18% |
| Recall (macro) | 88.91% |
| F1-score (macro) | 88.83% |
| Training Time | 106.6 s |
| Total Parameters | 2,426,725 |

**5-fold cross-validation comparison:**

| Configuration | Mean CV Accuracy (%) | SD |
|---------------|----------------------:|----:|
| C1_baseline | 91.09 | 0.63 |
| C2_best_reg | 90.76 | 0.51 |
| C3_best_opt_hparams | 91.09 | 0.74 |
| C4_fine_tuned | 91.25 | 1.20 |

**Optimizer comparison (6 epochs, lr = 0.001, frozen base):**

| Optimizer | Final Loss | Best Val. Accuracy (%) | Epoch to Converge | Time (s) |
|-----------|-----------:|-------------------------:|--------------------:|----------:|
| SGD | 3.1526 | 29.89 | 6 | 38.1 |
| Momentum | 0.8350 | 89.86 | 6 | 38.2 |
| RMSProp | 0.2386 | 92.21 | 4 | 38.5 |
| Adam | 0.2433 | 92.57 | 6 | 38.9 |

> **Note:** All four weight-initialization strategies (Zero, Random, Xavier, He)
> produced an identical flat ≈2.7% validation accuracy when training MobileNetV2
> from scratch for only 6 epochs — essentially random-guess level for 37 classes.
> This is reported as-is and is the practical motivation for using transfer
> learning in the rest of the experiment, rather than training from scratch.

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
Experiment_5_Deep_Learning_Lab.ipynb
```

Run all cells sequentially. In Colab, set `Runtime → Change runtime type → T4 GPU`
before running for reasonable training times. The Oxford-IIIT Pet dataset (~800 MB)
is downloaded directly from the official VGG mirror on first run.

---

## Author
Aishwarya Muthukumar
