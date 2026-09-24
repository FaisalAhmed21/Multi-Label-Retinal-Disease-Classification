# Multi-Label Retinal Disease Classification (CNN)

A Convolutional Neural Network (CNN)-based medical image analysis project that classifies retinal fundus photographs into **20 possible disease/finding categories simultaneously** (multi-label classification), using transfer learning, an ensemble of three CNN backbones, and Grad-CAM interpretability.


---

## Overview

Retinal photographs frequently show evidence of more than one disease at once (e.g. Diabetic Retinopathy alongside Media Haze), so this project is framed as **multi-label** classification rather than single-label classification — each of the 20 labels is treated as an independent yes/no decision (sigmoid output + binary cross-entropy), not a single softmax choice.

The model uses **transfer learning** with three ImageNet-pretrained CNN backbones (EfficientNetB0, ResNet50, DenseNet121), a two-phase training strategy (frozen backbone warm-up → full fine-tuning), a class-imbalance-aware focal loss, per-class threshold calibration, and Grad-CAM for visual interpretability.

## Dataset

**Multi-Label Retinal Diseases (MuReD) Dataset**
Rodriguez Rivera, Al-Marzouqi & Liatsis (2022) — [Mendeley Data, DOI: 10.17632/pc4mb3h8hz.1](https://doi.org/10.17632/pc4mb3h8hz.1)

- 2,208 retinal fundus images (mixed `.png` / `.tiff` formats, resolutions from 520×520 up to 3400×2800)
- 20 binary disease/finding labels per image (DR, NORMAL, MH, ODC, TSLN, ARMD, DN, MYA, BRVO, ODP, CRVO, CNV, RS, ODE, LS, CSR, HTR, ASR, CRS, OTHER)
- Built from ARIA, STARE, and RFMiD source datasets, with post-processing to reduce class imbalance relative to earlier retinal disease datasets

The dataset ships with only a train/validation split, so the validation set was further split 50/50 into a true validation set and a held-out test set:

| Split | Images |
|---|---|
| Train | 1,764 |
| Validation | 222 |
| Test | 222 |

## Model Architecture

Input (224x224x3)
-> Pretrained CNN Backbone (ImageNet weights)
-> GlobalAveragePooling2D
-> Dropout(0.3)
-> Dense(256, ReLU)
-> BatchNormalization
-> Dropout(0.3)
-> Dense(20, Sigmoid)

Three backbones were trained independently with this identical head, then combined via **soft-voting ensembling** (averaging predicted probabilities):

- **EfficientNetB0** — compound-scaled efficient convolutions
- **ResNet50** — residual connections
- **DenseNet121** — densely-connected feature reuse

**Training:** two-phase transfer learning per backbone — Phase 1 trains only the head with the backbone frozen (lr=1e-3), Phase 2 unfreezes and fine-tunes the entire backbone at a much lower learning rate (lr=1e-5). Loss is **Binary Focal Cross-Entropy** (class-balanced, γ=2.0) to address the severe class imbalance in the dataset. Early stopping, learning-rate reduction on plateau, and checkpointing were used throughout.

**Decision thresholds:** per-class thresholds were tuned on the validation set (Youden's J statistic), with a minimum-support guard — classes with fewer than 10 validation examples fall back to a standard 0.5 threshold, since per-class optimal-threshold search is unreliable on that little data.

## Results

### Single Model vs. Ensemble

| Metric | EfficientNet Only | 3-Model Ensemble |
|---|---|---|
| Macro-F1 | 0.2637 | 0.2542 |
| Micro-F1 | 0.3343 | 0.4596 |
| Hamming Loss (lower is better) | 0.1579 | 0.0980 |
| Macro-AUROC | 0.8268 | 0.8944 |

The ensemble measurably improved overall discrimination (Macro-AUROC) and reduced individual wrong-label predictions (Micro-F1, Hamming loss), but did **not** improve Macro-F1 — the metric that weights all 20 classes equally. This shows the ensemble helps on classes that already had reasonable support, but does not solve the underlying data-scarcity problem for the rarest classes (as few as 3–8 test examples), which is a dataset limitation rather than a model/architecture limitation.

### Per-Class Test Set Performance (Ensemble)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DR | 0.64 | 0.86 | 0.73 | 49 |
| NORMAL | 0.50 | 0.96 | 0.66 | 49 |
| MH | 0.37 | 0.65 | 0.47 | 17 |
| ODC | 0.27 | 0.88 | 0.42 | 26 |
| TSLN | 0.37 | 0.94 | 0.53 | 16 |
| ARMD | 0.18 | 0.88 | 0.29 | 16 |
| DN | 0.15 | 0.44 | 0.22 | 16 |
| MYA | 1.00 | 0.44 | 0.62 | 9 |
| RS | 1.00 | 0.17 | 0.29 | 6 |
| ODE | 0.67 | 0.40 | 0.50 | 5 |
| OTHER | 0.25 | 0.73 | 0.37 | 26 |
| BRVO / ODP / CRVO / CNV / LS / CSR / HTR / ASR / CRS | 0.00 | 0.00 | 0.00 | ≤8 each |
| **Macro avg** | **0.27** | **0.37** | **0.25** | 278 |
| **Micro avg** | **0.35** | **0.67** | **0.46** | 278 |

## Key Finding & Limitation

The 9 classes scoring 0.00 F1 each have fewer than 10 test examples — this is fundamentally a **data-scarcity problem**, not a model-capacity problem. No amount of additional backbone diversity or ensembling can manufacture signal that isn't present in a handful of training images. The most promising next steps are targeted oversampling/augmentation for the rarest classes, or sourcing additional labeled examples for them specifically.

## Interpretability: Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize which regions of each retinal image most influenced the model's predictions, confirming the model attends to actual retinal features rather than background artifacts. See the notebook for the generated heatmaps.

## Repository Structure

.
├── MuReD_Ensemble_GradCAM_FINAL.ipynb # Full notebook: EDA, preprocessing, training, evaluation, Grad-CAM
├── train_data.csv # Training labels
├── val_data.csv # Validation labels (split into val/test in the notebook)
├── report.pdf # Written report (architecture, methodology, results)
└── README.md


> Note: the `images/` folder (2,208 fundus images) is not included in this repository due to size — download it directly from the [MuReD dataset page on Mendeley Data](https://doi.org/10.17632/pc4mb3h8hz.1).

## How to Run

1. Download the MuReD dataset (`train_data.csv`, `val_data.csv`, and the `images/` folder) from [Mendeley Data](https://doi.org/10.17632/pc4mb3h8hz.1).
2. Open `MuReD_Ensemble_GradCAM_FINAL.ipynb` in Google Colab.
3. Upload the dataset to Google Drive (CSV files + `images/` folder in one project folder).
4. Update the `PROJECT_DIR` variable in the notebook's second cell to point to that folder.
5. Run all cells top to bottom (GPU runtime recommended: Runtime → Change runtime type → GPU).

## Tech Stack

- Python, TensorFlow / Keras
- EfficientNetB0, ResNet50, DenseNet121 (ImageNet pretrained)
- scikit-learn (metrics), pandas, seaborn/matplotlib (EDA & visualization)
- Google Colab + Google Drive (training environment & checkpoint persistence)
