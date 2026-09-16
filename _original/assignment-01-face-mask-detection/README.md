# Assignment 01 — Face Mask Detection (CNN + VGG19)

## Task Overview

Build and compare two image classifiers to detect whether a person is wearing a face mask. This was motivated by real-world COVID-era access control scenarios where AI can automate the check.

**Task type**: Binary image classification (with mask / without mask)
**Framework**: TensorFlow 2.13 / Keras
**Platform**: Google Colab (CPU — no GPU available at training time)

---

## Dataset

**Source**: [Kaggle — Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset/data)

| Split | Images |
|---|---|
| Train | 10,000 |
| Validation | 800 |
| Test | 992 |

- With-mask images: scraped from Google Search
- Without-mask images: pre-processed from the CelebFace dataset (Jessica Li)
- Class distribution: roughly balanced (~50/50)

---

## Approach

### Model 1 — Custom CNN (`cnn-face-mask-binary-classification.ipynb`)

Architecture (built from scratch):

```
Input (256×256×3)
→ Conv2D(16, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Conv2D(32, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Conv2D(32, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Conv2D(64, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Flatten → Dense(128, ReLU) → Dropout(0.2)
→ Dense(1, Sigmoid)
```

Total parameters: **1,638,721** (~6.25 MB)

Training config:
- Optimizer: Adam (lr=1e-4)
- Loss: Binary Cross-Entropy
- Batch size: 16, Max epochs: 32
- Early stopping: patience=3, monitor=val_loss
- Data augmentation: rescale, shear, zoom, samplewise normalization

### Model 2 — VGG19 Transfer Learning (`vgg19-transfer-learning-face-mask-detection.ipynb`)

- Base: VGG19 pretrained on ImageNet (frozen)
- Custom head appended for binary classification

---

## Training Results

### Custom CNN

| Epoch | Train Loss | Val Loss | Val Accuracy |
|---|---|---|---|
| 1 | 0.1955 | 0.1671 | 96.00% |
| 5 | 0.0388 | 0.0380 | 98.75% |
| 10 | 0.0184 | 0.0135 | 99.50% |
| 16 | 0.0070 | 0.0119 | 99.62% |
| 20 | 0.0038 | 0.0076 | **99.75%** (best) |
| 23 | — | — | stopped (EarlyStopping) |

Final test set evaluated with confusion matrix — very few misclassifications.

---

## Inference Demo

The notebook includes an OpenCV-based pipeline that:
1. Loads the saved `base_model.h5`
2. Uses Haar Cascade to detect faces in an input image
3. Crops each face → resizes to 256×256 → predicts mask/no-mask
4. Draws color-coded bounding box and label on the image

```python
mask_label = {0: 'WITH MASK', 1: 'NO MASK'}
shape_color = {0: (255,0,0), 1: (0,0,255)}   # blue=mask, red=no mask
```

---

## File Structure

```
assignment-01-face-mask-detection/
├── F112119106/
│   ├── cnn-face-mask-binary-classification.ipynb   # main notebook (student work)
│   ├── F112119106.pdf                              # submitted report
│   └── F112119106.zip                             # submission archive
├── face-mask-detection-main/
│   └── face-mask-detection-main/
│       ├── notebooks/
│       │   ├── cnn-face-mask-detection-training.ipynb      # reference CNN
│       │   └── vgg19-transfer-learning-face-mask-detection.ipynb  # VGG19
│       ├── images/                  # training/evaluation plots
│       └── README.md               # original project README
└── report.docx                     # assignment report
```

---

## How to Run

1. Open either notebook in Google Colab or Jupyter
2. Upload your Kaggle API key when prompted (`files.upload()`)
3. Dataset will auto-download and unzip (~330MB)
4. Run all cells top to bottom
5. Trained model saved as `base_model.h5`

> **Note**: Training on CPU takes ~14 minutes per epoch (23 epochs total). Enable GPU runtime in Colab to reduce to ~1–2 min/epoch.

---

## Key Observations

- The custom CNN converged well and reached near-perfect accuracy without transfer learning.
- EarlyStopping fired at epoch 23 (patience=3 after best at epoch 20).
- With samplewise normalization, the model was robust across lighting conditions.
- VGG19 converges faster due to pretrained features but requires more memory.

---

## References

- Dataset: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
- Reference project: https://github.com/tajammulbasheer/face-mask-detection
