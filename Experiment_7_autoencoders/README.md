# Autoencoders-CAE-DAE-VAE-Image-Reconstruction
End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders for Image Representation, Reconstruction and Generation

## Overview
This repository contains the implementation of an **end-to-end study of autoencoders
and their variants** using **TensorFlow/Keras** as part of the
**CS3807 – Deep Learning Laboratory (Experiment 7)**.

The experiment demonstrates the complete pipeline for image representation learning,
reconstruction, denoising and generative modeling, including:
- Image preparation from the MNIST handwritten digit dataset
- A fully connected autoencoder (FC-AE) with a 16-dimensional latent code
- A Convolutional Autoencoder (CAE) preserving spatial structure
- A Denoising CAE trained on Gaussian-corrupted images
- A Variational Autoencoder (VAE) with a 2-D latent space, reparameterization trick,
  sampling and latent interpolation
- Quantitative evaluation with MSE, MAE and SSIM, and qualitative analysis through
  reconstructions, error maps, latent-space plots and generated samples
- Additional exercises: latent-dimension sweeps, noise-type comparison,
  transposed-convolution decoder, VAE latent size, β-VAE and larger sample grids

---

## Objective
To develop an end-to-end understanding of autoencoders by implementing a fully connected
autoencoder, a convolutional autoencoder, a denoising autoencoder and a variational
autoencoder on the same dataset, comparing deterministic and probabilistic latent
representations, evaluating reconstruction quality, and exploring the generative
capability of the VAE.

---

## Dataset

### MNIST Handwritten Digits
**Source:** https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
(loaded directly via `tf.keras.datasets.mnist`)

- Grayscale images of digits 0–9, each of shape `28 × 28 × 1`
- Pixel values scaled from `[0, 255]` to `[0, 1]`
- Laboratory subset: the first **10,000 training images** and the first **2,000 test images**
- Labels are **not** used as targets (the image itself is the target); they are used
  only to colour the VAE latent-space plot

| Array | Training | Test |
|-------|---------:|-----:|
| Raw images | (10000, 28, 28) | (2000, 28, 28) |
| Flattened (FC-AE) | (10000, 784) | (2000, 784) |
| With channel axis (CAE, DAE, VAE) | (10000, 28, 28, 1) | (2000, 28, 28, 1) |

The test images were used only as validation data to monitor the loss; they never
updated the weights or selected a model.

---

## Repository Contents

| File | Description |
|------|-------------|
| Experiment_7_Autoencoders.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| outputs/ | Generated plots and result tables |
| Experiment_7_Report.pdf | Full lab report |

---

## Experiments Performed

### Section 7–9
Fully Connected Autoencoder
- Architecture: `784 → 128 → 32 → 16 → 32 → 128 → 784` (ReLU hidden layers, sigmoid output)
- Optimizer: Adam, lr = 1e-3, batch size 128, 20 epochs, binary cross-entropy
- 211,040 parameters
- Original vs. reconstruction grid, absolute error maps, training/validation loss
- Test MSE, MAE and SSIM

### Section 10–11
Convolutional Autoencoder and FC-AE vs. CAE Comparison
- `Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(64) → UpSampling → Conv(32) → UpSampling → Conv(1, sigmoid)`
- 74,497 parameters; latent feature map of 7×7×64 = 3,136 values
- Matched-latent-size comparison (d = 8, 16, 32) to separate the effect of convolution
  from the effect of a larger bottleneck

### Section 12–14
Denoising Autoencoder
- Same CAE architecture, trained on Gaussian-noisy input (σ = 0.2) with clean targets
- Evaluated on Gaussian noise (σ = 0.1, 0.2, 0.3) and unseen salt-and-pepper noise
  (p = 0.05, 0.10, 0.20)
- Clean / noisy / denoised grid and noise level vs. MSE/MAE/SSIM plots

### Section 15–20
Variational Autoencoder
- 2-D latent space; encoder outputs μ and log σ²; reparameterization trick z = μ + σ ⊙ ε
- Loss = summed binary cross-entropy + closed-form KL divergence to N(0, I)
- 485,957 parameters; Adam, lr = 1e-3, batch size 128, 20 epochs
- Latent-space scatter plot, 5×5 generated samples, and latent interpolation

### Section 21–25
Cross-Model Evaluation
- Consolidated MSE / MAE / SSIM / parameter count / training time for all models
- Reconstruction-error histogram and analysis of the five highest-error test images

### Section 26–27
Required Inferences and Latent-Dimension Study
- FC-AE repeated for d ∈ {2, 8, 16, 32}
- Ten mandatory inferences consolidated

### Section 29
Additional Exercises
- CAE with a dense bottleneck for d ∈ {4, 8, 16, 32}
- Gaussian vs. salt-and-pepper noise
- Noise-level generalisation of the denoising model
- Transposed-convolution decoder vs. up-sampling decoder
- VAE with d = 2 vs. d = 8
- β-VAE (β = 1 vs. β = 2)
- 100-sample VAE grid and digit 0 → digit 1 interpolation

---

## Results

**Reconstruction on 2,000 test images:**

| Model | MSE | MAE | SSIM | Parameters | Time (s) |
|-------|----:|----:|-----:|-----------:|---------:|
| FC Autoencoder | 0.022668 | 0.059465 | 0.728743 | 211,040 | ≈9 |
| Conv. Autoencoder | 0.002580 | 0.014898 | 0.974509 | 74,497 | ≈31 |
| Denoising CAE (σ = 0.2) | 0.004471 | 0.020788 | 0.946785 | 74,497 | ≈26 |
| VAE (d = 2) | 0.044742 | 0.103395 | 0.467958 | 485,957 | ≈28 |

**Matched latent size (dense bottleneck of d values):**

| d | FC-AE MSE | CAE MSE | FC-AE SSIM | CAE SSIM |
|--:|----------:|--------:|-----------:|---------:|
| 8 | 0.026644 | 0.022872 | 0.687447 | 0.744092 |
| 16 | 0.022865 | 0.012497 | 0.730126 | 0.863173 |
| 32 | 0.020106 | 0.008689 | 0.758167 | 0.905526 |

**Denoising performance (Gaussian noise, model trained at σ = 0.2):**

| σ | MSE | MAE | SSIM |
|--:|----:|----:|-----:|
| 0.1 | 0.003409 | 0.017357 | 0.960372 |
| 0.2 | 0.004472 | 0.020807 | 0.946717 |
| 0.3 | 0.007147 | 0.028827 | 0.884323 |

**FC-AE latent dimension study:**

| Latent Dim | MSE | SSIM |
|-----------:|----:|-----:|
| 2 | 0.055202 | 0.317665 |
| 8 | 0.026644 | 0.687447 |
| 16 | 0.022865 | 0.730126 |
| 32 | 0.020106 | 0.758167 |

**VAE (final epoch):** reconstruction loss 150.14, KL 5.62, total 155.76 (training);
151.29 / 5.59 / 156.88 (validation).

> **Note:** The CAE's 7×7×64 latent map is larger than the 784-pixel input, so part of its
> advantage over the FC-AE comes from the absence of a compressive bottleneck; the
> matched-size table shows convolution still helps in itself (about 1.2–2.3× lower MSE).
> All numbers come from single training runs without a fixed random seed, so small
> differences (e.g. between the two decoder types) lie within run-to-run variation.
> VAE losses are per-image sums and are not on the same scale as MSE/MAE. The denoising
> model was trained at σ = 0.2 only, so other noise levels and salt-and-pepper noise
> test generalisation.

---

## Dependencies
See `requirements.txt`.

---

## Execution Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Lunarya-git/Deep-Learning-Experiments.git
cd Deep-Learning-Experiments/Experiment_7_autoencoders
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
Experiment_7_Autoencoders.ipynb
```

Run all cells sequentially. In Colab, set `Runtime → Change runtime type → T4 GPU`
for faster training. MNIST is downloaded automatically through Keras, so no manual
data setup is needed.

---

## Author
Aishwarya Muthukumar
