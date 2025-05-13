# Distillation Benchmarks

## Synopsis

This project is a modular framework for benchmarking knowledge distillation (KD) techniques across a variety of convolutional neural networks and datasets. It supports multiple popular distillation strategies including classic logit-based distillation, Factor Transfer (FT), and Tucker decomposition-based distillation. 

By unifying these strategies under a single codebase with consistent evaluation and logging, the project allows direct performance comparisons of student-teacher model pairs under controlled and reproducible conditions.

---

## Deeper Explanation

The core idea behind knowledge distillation is to transfer the generalization ability of a large "teacher" model to a smaller "student" model. This repository implements three distinct approaches:

- **Standard Knowledge Distillation (KD)**: A baseline method where the student mimics softened logits from the teacher in addition to learning from ground truth.
- **Factor Transfer (FT)**: A more nuanced approach where a "paraphraser" is first trained to extract meaningful factors from the teacher's feature maps, and a "translator" is then trained alongside the student to replicate those factors.
- **Tucker-Based Distillation (TD)**: Introduces a low-rank tensor decomposition (Tucker) on the spatial feature maps to compress intermediate representations. The student is trained to match this compressed view of the teacher’s activations.

All approaches follow a common evaluation pattern and logging pipeline, producing training/testing curves, final checkpoints, and pickled logs for further analysis.

---

## Experiment Variants and Details

### 1. **Vanilla Supervised Training**
Used as a control, the student model is trained from scratch using only cross-entropy loss on ground truth labels. It includes label smoothing for improved generalization.

### 2. **Logit-Based Knowledge Distillation (KD)**
Implements temperature-scaled soft-target learning where the student optimizes a combination of:
- KL-divergence between softmax outputs of teacher and student (with temperature scaling), and
- Cross-entropy loss with true labels.

This combination balances hard and soft target supervision.

### 3. **Factor Transfer (FT)**
- **Paraphraser**: First trained independently on teacher feature maps to compress them into a compact representation.
- **Translator**: Learns to reconstruct this compressed representation from student features.
- Final loss is a combination of L1 loss between factor representations and classification loss.
- Visualizations track paraphraser loss and student accuracy during training.

### 4. **Tucker Decomposition Distillation (TD)**
- Applies Tucker decomposition on teacher and student intermediate feature maps.
- Loss is computed on normalized representations of the decomposed core tensors.
- Supports both full-tensor and sample-wise decomposition.
- Helps the student match teacher's abstracted feature encoding rather than raw values.

---

## Automation and Configuration

Experiments are launched through a PBS/SLURM-compatible `runner.py` script. Parameters such as dataset, model architecture, distillation type, and experiment IDs are customizable per run.

Each experiment automatically:
- Creates unique folders for logs and weights
- Saves performance plots (`Loss.png`, `Accuracy.png`)
- Stores results in serialized `.pkl` logs
- Dumps final student weights to disk

---

## Notebook Explorations

A series of scratchpad notebooks accompany this project to support experimental analysis, feature map probing, and metric development:

### `scratchpad.ipynb` & `scratchpad_2.ipynb`
These notebooks are primarily exploratory environments where custom feature map metrics are developed and tested. The experiments focus on analyzing intermediate CNN activations to understand their statistical properties and behavior across student and teacher networks.

They utilize metrics such as:
- **Wavelet-based high-frequency energy**
- **Frequency-domain entropy**
- **Skewness, kurtosis, and IQR**
- **Low-pass residual energy**
- **Coefficient of variation and standard deviation**

These metrics are intended to capture differences in how student and teacher networks encode feature representations—beyond standard accuracy-based evaluations.

### `scratchpad_3.ipynb` & `scratchpad_4.ipynb`
These focus more on visualizing and comparing feature distributions before and after distillation. They include:
- Heatmaps and Fourier transforms of activation channels
- Histogram comparisons of student vs teacher feature maps
- Metric-driven analyses to select layers and channels most affected by training

These notebooks serve as diagnostic and interpretability tools for understanding what is being preserved, transferred, or lost during distillation.

Combined with `noise_metrics.py`, these notebooks provide a toolkit for quantifying information flow and compression in neural feature spaces.

---
