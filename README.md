# ✍️ Handwritten Character Recognition
### CodeAlpha Machine Learning Internship — Task 3

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-red?style=flat-square&logo=keras)
![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![GPU](https://img.shields.io/badge/GPU-2×%20Tesla%20T4-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)

---

## 📌 Objective

Build a **Convolutional Neural Network (CNN)** that recognizes handwritten characters — both digits (0–9) and uppercase letters (A–Z) — from 28×28 grayscale images. The model is trained on a combined dataset of **148,800 images** across **36 classes**.

---

## 📂 Datasets

| Dataset | Source | Classes | Train Images | Test Images |
|---|---|---|---|---|
| **MNIST** | `keras.datasets.mnist` | Digits 0–9 | 60,000 | 10,000 |
| **EMNIST Letters** | `tensorflow_datasets` (`emnist/letters`) | Letters A–Z | 88,800 | 14,800 |
| **Combined** | Merged via numpy | **36 total** | **148,800** | **24,800** |

### Label space

| Label range | Characters | Source |
|---|---|---|
| `0 – 9` | Digits 0–9 | MNIST (original labels unchanged) |
| `10 – 35` | Letters A–Z | EMNIST (remapped from 1–26 → 10–35) |

---

## 🔍 Project Pipeline

```
MNIST + EMNIST → Combine → Normalize → Reshape → Build CNN → Train → Save → Evaluate
```

### 1. Load datasets
- MNIST: `keras.datasets.mnist` — built-in, loads instantly, no download
- EMNIST: `tensorflow_datasets.load('emnist/letters')` — downloads ~535MB automatically on first run, cached for subsequent runs

### 2. Preprocessing
- **Orientation fix:** EMNIST images are stored rotated 90° — corrected with `np.transpose(X, (0,2,1))`
- **Label remapping:** EMNIST labels 1–26 → 10–35 using `y + 9` to avoid overlap with MNIST digits 0–9
- **Combine:** `np.concatenate` merges both datasets along axis 0
- **Normalize:** pixel values `/ 255.0` → range 0.0 to 1.0
- **Reshape:** `(N, 28, 28)` → `(N, 28, 28, 1)` — adds channel dimension required by Conv2D

### 3. CNN Architecture (3-block design)

| Layer | Output shape | Parameters |
|---|---|---|
| `Input` | (28, 28, 1) | 0 |
| `Conv2D(32, 3×3, relu)` | (26, 26, 32) | 320 |
| `MaxPooling2D(2×2)` | (13, 13, 32) | 0 |
| `Conv2D(64, 3×3, relu)` | (11, 11, 64) | 18,496 |
| `MaxPooling2D(2×2)` | (5, 5, 64) | 0 |
| `Conv2D(128, 3×3, relu)` | (3, 3, 128) | 73,856 |
| `Flatten` | (1152,) | 0 |
| `Dense(256, relu)` | (256,) | 295,168 |
| `Dropout(0.4)` | (256,) | 0 |
| `Dense(128, relu)` | (128,) | 32,896 |
| `Dropout(0.3)` | (128,) | 0 |
| `Dense(36, softmax)` | (36,) | 4,644 |
| **Total** | | **425,380** |

### 4. Training configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Sparse categorical crossentropy |
| Epochs | 10 |
| Batch size | 128 |
| Validation split | 10% |
| Hardware | 2× Tesla T4 GPU (Kaggle) |

### 5. Model saving
Trained model saved with `model.save('cnn_handwriting_model.keras')` immediately after training — allows reloading without retraining.

### 6. Evaluation
- Test accuracy and loss on 24,800 unseen images
- Confusion matrix heatmap across all 36 classes
- Full classification report — precision, recall, F1 per class
- Visual prediction grid — 16 random test images (green = correct, red = wrong)
- Single character prediction with top-5 confidence scores

---

## 📊 Results

| Metric | Value |
|---|---|
| **Test accuracy** | **95.00%** |
| **Test loss** | **0.1501** |
| Train accuracy (epoch 10) | ~96.6% |
| Val accuracy (epoch 10) | ~93.5% |

### Training history

| Epoch | Train accuracy | Val accuracy |
|---|---|---|
| 1 | 78.2% | 90.0% |
| 2 | 91.8% | 91.4% |
| 3 | 93.6% | 92.2% |
| 4 | 94.5% | 93.1% |
| 5 | 95.0% | 92.8% |
| 10 | ~96.6% | ~93.5% |

### Per-class performance (classification report)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.98 | 1.00 | 0.99 | 980 |
| 1 | 1.00 | 0.99 | 0.99 | 1135 |
| 2 | 1.00 | 0.99 | 0.99 | 1032 |
| 3 | 0.99 | 1.00 | 0.99 | 1010 |
| 4 | 0.99 | 0.99 | 0.99 | 982 |
| 5 | 0.99 | 0.98 | 0.99 | 892 |
| 6 | 1.00 | 0.98 | 0.99 | 958 |
| 7 | 0.98 | 1.00 | 0.99 | 1028 |
| 8 | 0.99 | 1.00 | 0.99 | 974 |
| 9 | 0.99 | 0.98 | 0.99 | 1009 |
| A | 0.95 | 0.93 | 0.94 | 800 |
| B | 0.97 | 0.97 | 0.97 | 800 |
| C | 0.98 | 0.96 | 0.97 | 800 |
| D | 0.96 | 0.95 | 0.96 | 800 |

> ✅ Digits achieve near-perfect F1 (~0.99). Letters average ~0.94–0.97 — slightly lower due to visual similarity between characters (e.g. 0 vs O, 1 vs I, 5 vs S).

### Single prediction example
```
Predicted: '4'   Actual: '4'   Confidence: 99.99%
Top 5: 4 (99.99%), 7 (0.00%), 1 (0.00%), Y (0.00%), 9 (0.00%)
```

---

## 🗂️ Repository Structure

```
CodeAlpha_HandwrittenRecognition/
│
├── handwrittenrecognition-codealpha.ipynb   # Main Kaggle notebook
├── cnn_handwriting_model.keras              # Saved trained model weights
└── README.md                               # This file
```

---

## ▶️ How to Run

### On Kaggle (recommended)
1. Open the notebook on Kaggle
2. Enable GPU: Settings → Accelerator → **GPU T4 x2**
3. Click **Run All** — MNIST loads instantly, EMNIST downloads automatically on first run (~535MB, ~1 min), cached after that
4. Total runtime: ~6 minutes with GPU

### Locally
```bash
# Clone the repo
git clone https://github.com/your-username/CodeAlpha_HandwrittenRecognition.git
cd CodeAlpha_HandwrittenRecognition

# Install dependencies
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn scikit-learn

# Launch notebook
jupyter notebook handwrittenrecognition-codealpha.ipynb
```

### Load saved model (skip retraining)
```python
from tensorflow import keras
model = keras.models.load_model('cnn_handwriting_model.keras')
```

---

## 📓 Notebook Structure

| Cell | Contents |
|---|---|
| Cell 1 | Imports — TensorFlow 2.19, Keras, tfds, sklearn |
| Cell 2 | Load MNIST — 60k train / 10k test, visualize 8 digits |
| Cell 3 | Load EMNIST via tfds — 88.8k train / 14.8k test, fix orientation, remap labels 1–26 → 10–35 |
| Cell 4 | Combine datasets — normalize, reshape to (N,28,28,1), build LABEL_MAP |
| Cell 5 | Build CNN — 3 conv blocks + dense classifier, `model.summary()` |
| Cell 6 | Compile + train — Adam, sparse_categorical_crossentropy, 10 epochs |
| Cell 7 | Save model — `cnn_handwriting_model.keras` |
| Cell 8 | Plot training history — accuracy + loss curves |
| Cell 9 | Evaluate — test accuracy 95.00%, confusion matrix, classification report |
| Cell 10 | Prediction grid — 16 random test images, green/red labels |
| Cell 11 | Verify label map — actual vs expected label ranges |
| Cell 12 | Single prediction — one character with top-5 confidence scores |

---

## 🧠 Key Findings

- **3-block CNN** (32→64→128 filters) achieves **95.00% accuracy** across 36 classes (digits + letters) in ~6 minutes on GPU.
- **EMNIST requires a transpose fix** — images are stored rotated 90° vs MNIST. Without `np.transpose(X, (0,2,1))`, the model trains on malformed letter images.
- **Label remapping is essential** — EMNIST uses labels 1–26 which overlap with MNIST digits. Remapping to 10–35 creates a clean non-overlapping 36-class label space.
- **Digits outperform letters** (~99% vs ~94–97% F1) — each digit class has 6,000 training images vs only ~3,415 per letter class on average.
- **Dropout regularization** (0.4 + 0.3) keeps overfitting minimal — train accuracy 96.6% vs val 93.5%.
- **GPU acceleration** (2× Tesla T4) cuts epoch time from ~60s to ~5s, making 10-epoch training practical.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **TensorFlow 2.19 / Keras 3.x** — CNN architecture, training, inference, model saving
- **TensorFlow Datasets (tfds)** — EMNIST letters loading and caching
- **NumPy** — array operations, dataset concatenation, label remapping
- **Matplotlib / Seaborn** — training curves, confusion matrix heatmap, prediction grids
- **Scikit-learn** — confusion matrix, classification report
- **Kaggle Notebooks** — GPU-accelerated cloud environment (2× Tesla T4, 13,757MB each)

---

## 👤 Author

**Vishnu Prasad G S (VP)**
CodeAlpha Machine Learning Intern
📍 Coimbatore, India

[![GitHub](https://img.shields.io/badge/GitHub-your--username-black?style=flat-square&logo=github)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)

---

## 🏢 Internship

This project was completed as **Task 3** of the **CodeAlpha Machine Learning Internship**.

🌐 [www.codealpha.tech](https://www.codealpha.tech)
