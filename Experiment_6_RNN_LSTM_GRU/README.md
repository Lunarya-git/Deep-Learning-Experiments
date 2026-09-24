# RNN-LSTM-GRU-Sequence-Learning-Video-Understanding
End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

## Overview
This repository contains the implementation of an **end-to-end study of recurrent
sequence learning** using **TensorFlow/Keras** as part of the
**CS3807 – Deep Learning Laboratory (Experiment 6)**.

The experiment demonstrates the complete pipeline for sequence classification and
video understanding, including:
- Sequential data preparation from the UCI Human Activity Recognition (HAR) dataset
- Backpropagation Through Time (BPTT) — conceptual explanation and a hand-verified
  numerical example
- Implementation and comparison of Vanilla RNN, LSTM and GRU for activity
  classification
- Training/validation curve analysis and confusion-matrix analysis
- Effect of sequence length (T = 32, 64, 128) on RNN/LSTM/GRU performance
- CNN feature extraction (pretrained MobileNetV2) + LSTM/GRU for video action
  recognition on a UCF101 subset
- A synthetic sequence-to-sequence (reversal) task using an LSTM encoder–decoder
  with teacher forcing and greedy decoding
- Additional exercises: recurrent-unit sweep, stacked recurrent layers,
  bidirectional vs. unidirectional LSTM, and a variable-length seq2seq task

---

## Objective
To develop an end-to-end understanding of recurrent sequence learning by
implementing and comparing Vanilla RNN, LSTM and GRU on smartphone sensor data,
extending the pipeline to video understanding using CNN-extracted features with
recurrent networks, and demonstrating the encoder–decoder framework on a
sequence-to-sequence task.

---

## Datasets

### 1. UCI Human Activity Recognition Using Smartphones (primary dataset)
**Source:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Raw inertial signal files (body acceleration, body gyroscope and total
acceleration — 9 channels total) were used, rather than the precomputed
561-feature vectors, to preserve full temporal structure.

- Input representation: `X ∈ R^(N × 128 × 9)` (128 time steps, 9 channels)
- 6 activity classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING,
  STANDING, LAYING
- Laboratory subset: 3,000 windows drawn from the full pool of 10,299
  (500 windows per class, balanced)
- Stratified 70/15/15 train/validation/test split

| Split | Windows | Per class |
|-------|--------:|----------:|
| Training | 2,100 | 350 |
| Validation | 450 | 75 |
| Testing | 450 | 75 |

### 2. UCF101 (video understanding subset)
**Source:** Khurram Soomro et al., "UCF101: A Dataset of 101 Human Actions Classes
From Videos in the Wild," 2012.

A small subset restricted to 3 action classes — `IceDancing`, `WalkingWithDog`,
`YoYo` — was used (45 videos found, 43 usable after discarding videos from which
10 frames could not be reliably extracted). Split 70/15/15 (stratified) into
30 training, 6 validation and 7 test videos. Each video: 10 uniformly sampled
frames, resized to 224×224×3.

### 3. Synthetic sequence reversal dataset (seq2seq task)
8,000 random sequences of length 4 (digits 1–9), e.g. `[1,4,7,2] → [2,7,4,1]`.
Split into 6,800 training / 1,200 test sequences, with 10% of training (680)
used for validation.

---

## Repository Contents

| File | Description |
|------|-------------|
| Experiment_6_RNN_LSTM_GRU.ipynb | Complete implementation |
| requirements.txt | Python dependencies |
| README.md | Project documentation |
| outputs/ | Generated plots and result CSVs |
| Experiment_6_Report.pdf | Full lab report |

---

## Experiments Performed

### Section 7–8
Vanilla RNN Architecture and Backpropagation Through Time
- Hidden-state update equation derived and explained
- Vanishing/exploding gradient challenges discussed
- Hand-calculated numerical example verified against program output

### Section 9
RNN Implementation
- `SimpleRNN(32) → Dropout(0.2) → Dense(16, ReLU) → Dense(6, Softmax)`
- Optimizer: Adam, lr = 1e-3, batch size 32, 30 epochs

### Section 10–12
LSTM and GRU Architecture and Implementation
- Forget/input/output gates and cell-state update derived for LSTM
- Update/reset gates and candidate hidden state derived for GRU
- Same classifier head reused for RNN, LSTM and GRU (only the recurrent layer
  swapped) so the three-way comparison is controlled

### Section 13–15
Training Curves, Performance Evaluation and Confusion Matrix Analysis
- Training/validation loss and accuracy compared across all three models
- Test-set accuracy, macro precision/recall/F1 computed once on the untouched
  test set
- Confusion matrices analyzed for class-wise and pairwise error patterns

### Section 16
RNN vs. LSTM vs. GRU Comparison
- Parameter count, training time and test F1-score compared as a
  performance/complexity/cost trade-off

### Section 17
Effect of Sequence Length
- Experiment repeated for T ∈ {32, 64, 128} with identical splits and
  normalization statistics

### Section 18–21
Video Understanding (CNN–LSTM / CNN–GRU)
- Frozen pretrained MobileNetV2 used as a per-frame spatial feature extractor
  (1280-d feature vector per frame)
- 10×1280 feature sequence per video fed into LSTM/GRU for action
  classification

### Section 22–24
Sequence-to-Sequence Learning
- Encoder–decoder LSTM (64 units each) trained on the synthetic reversal task
  with teacher forcing
- Evaluated with greedy decoding using token accuracy and sequence accuracy

### Section 28
Additional Exercises
- Effect of recurrent units (16 vs. 32 vs. 64)
- Stacked (2-layer) recurrent models vs. 1-layer baselines
- Bidirectional vs. unidirectional LSTM
- Sequence-to-sequence task with a different output length (5 → 3)

---

## Results

**HAR test-set performance (450 windows, 32 units, single recurrent layer):**

| Metric | RNN | LSTM | GRU |
|--------|----:|-----:|----:|
| Accuracy (%) | 80.67 | 93.56 | 95.33 |
| Macro Precision (%) | 80.79 | 93.66 | 95.33 |
| Macro Recall (%) | 80.67 | 93.56 | 95.33 |
| Macro F1 (%) | 80.68 | 93.55 | 95.32 |
| Parameters | 1,974 | 6,006 | 4,758 |
| Training Time (s) | 29.64 | 29.51 | 24.69 |

**Effect of sequence length (test macro F1, %):**

| T | RNN | LSTM | GRU |
|---|----:|-----:|----:|
| 32 | 82.66 | 93.54 | 93.33 |
| 64 | 78.97 | 93.76 | 95.10 |
| 128 | 76.44 | 95.54 | 94.65 |

**Video action recognition (CNN–LSTM / CNN–GRU, 7 test videos):**

| Model | Accuracy | Precision | Recall | F1 | Trainable Parameters |
|-------|---------:|----------:|-------:|---:|----------------------:|
| CNN–LSTM | 100.00% | 100.00% | 100.00% | 100.00% | 168,643 |
| CNN–GRU | 100.00% | 100.00% | 100.00% | 100.00% | 126,723 |

**Sequence-to-sequence (reversal task, 1,200 test sequences):**

| Metric | Value |
|--------|-------|
| Token Accuracy | 100.00% |
| Sequence Accuracy | 100.00% |
| Final Training Loss | 0.0008 |
| Final Validation Loss | 0.0009 |
| Trainable Parameters | 51,083 |

> **Note:** The plain RNN's performance degrades as sequence length increases
> (82.66% → 76.44% F1 from T=32 to T=128), the clearest evidence of the
> vanishing-gradient problem in this experiment, while LSTM and GRU remain flat
> or improve slightly. All reported numbers come from single training runs;
> the video and reversal-task results are on very small test sets (7 videos,
> a length-4 task) and should be interpreted accordingly.

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
Experiment_6_RNN_LSTM_GRU.ipynb
```

Run all cells sequentially. In Colab, set `Runtime → Change runtime type → T4 GPU`
before running for reasonable training times. The UCI HAR dataset is downloaded
directly from the official source, and the UCF101 video subset is expected to be
mounted from Google Drive (`from google.colab import drive`) on first run.

---

## Author
Aishwarya Muthukumar
