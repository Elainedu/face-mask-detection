# Face Mask Detection

> 使用 CNN 和 VGG19 進行二元口罩偵測分類

## 📌 專案概述

本專案開發了兩個深度學習模型來偵測圖片中的人是否配戴口罩，這在 COVID-19 疫情期間對於自動化門禁控制和公共衛生管理具有實際應用價值。

## 🎯 專案目標

- 實作從零開始的 CNN 模型
- 應用遷移學習技術 (VGG19)
- 達到高準確率的二元分類
- 整合人臉偵測實現端到端系統

## 🛠️ 技術棧

- **Framework**: TensorFlow 2.13 / Keras
- **模型**: Custom CNN, VGG19 (ImageNet pretrained)
- **Dataset**: [Kaggle Face Mask 12K Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- **Tools**: OpenCV, Gradio

## 📊 主要成果

| 模型 | 準確率 | 參數量 | 訓練時間 |
|------|--------|--------|----------|
| **Custom CNN** | **99.75%** | 1.6M | 23 epochs (~14 min/epoch CPU) |
| **VGG19 Transfer Learning** | 99.6%+ | - | 收斂更快但記憶體需求更高 |

**關鍵指標:**
- Validation Accuracy: **99.75%**
- Final Loss: 0.0076
- 使用 EarlyStopping (patience=3)
- Data Augmentation: shear, zoom, samplewise normalization

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 訓練模型

```bash
# 訓練 CNN 模型
python train.py --model cnn --epochs 32

# 訓練 VGG19 模型
python train.py --model vgg19 --epochs 20
```

### 推理測試

```bash
# 單張圖片推理
python inference.py --image test.jpg --model results/models/cnn_best_model.h5

# 人臉偵測 + 口罩分類
python inference.py --image group.jpg --mode face-detection
```

### 啟動 Gradio Demo

```bash
python demo.py

# 或指定模型
python demo.py --model results/models/vgg19_best_model.h5
```

Demo 將在 http://localhost:7860 啟動

## 📁 專案結構

```
face-mask-detection/
├── train.py              # 訓練腳本
├── inference.py          # 推理腳本
├── demo.py              # Gradio 互動介面
├── requirements.txt     # 依賴套件
├── notebooks/           # Jupyter notebooks
│   ├── cnn-training.ipynb
│   └── vgg19-transfer-learning.ipynb
├── src/                 # 核心程式碼 (模組化)
├── data/                # 資料集 (需手動下載)
└── results/             # 訓練結果
    ├── models/          # 儲存的模型
    └── plots/           # 訓練曲線圖
```

## 🔬 方法論

### 模型 1: Custom CNN

6層卷積神經網路，從零開始訓練：

```
Input (256×256×3)
→ Conv2D(16, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Conv2D(32, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Conv2D(32, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Conv2D(64, 3×3, ReLU) → MaxPool → Dropout(0.2)
→ Flatten → Dense(128, ReLU) → Dropout(0.2)
→ Dense(1, Sigmoid)
```

**訓練配置:**
- Optimizer: Adam (lr=1e-4)
- Loss: Binary Cross-Entropy
- Batch Size: 16
- Data Augmentation: rescale, shear, zoom, samplewise normalization

### 模型 2: VGG19 Transfer Learning

- 使用 ImageNet 預訓練的 VGG19 作為特徵提取器
- 凍結基礎層，只訓練自定義的頂層分類器
- 收斂速度更快，但記憶體需求較高

## 📈 實驗結果

### CNN 訓練過程

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1     | 0.1955     | 0.1671   | 96.00%       |
| 5     | 0.0388     | 0.0380   | 98.75%       |
| 10    | 0.0184     | 0.0135   | 99.50%       |
| 20    | 0.0038     | **0.0076** | **99.75%**   |
| 23    | (Early Stopping) | -  | -            |

![Training Curves](results/plots/cnn_training_history.png)

## 💡 關鍵洞察

1. **Data Augmentation 關鍵**: Samplewise normalization 使模型對不同光線條件更穩健
2. **無需遷移學習**: Custom CNN 在此任務上已經達到極高準確率
3. **EarlyStopping 有效**: 在 epoch 20 達到最佳，自動在 epoch 23 停止
4. **人臉偵測整合**: 結合 Haar Cascade 實現端到端的實用系統

## 📦 資料集

**來源**: [Kaggle Face Mask 12K Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

| Split | Images |
|-------|--------|
| Train | 10,000 |
| Val   | 800    |
| Test  | 992    |

**類別分佈**: 約 50/50 平衡

**下載方式**:
1. 前往 Kaggle 下載資料集
2. 解壓縮到 `data/` 目錄
3. 確保結構為 `data/train/with_mask/`, `data/train/without_mask/` 等

## 🎨 Demo 畫面

Gradio 介面提供：
- 📤 圖片上傳
- 👤 自動人臉偵測
- 😷 口罩/無口罩分類
- 📊 信心度顯示
- 🎨 視覺化標註 (綠框=有口罩, 紅框=無口罩)

## 🔗 參考資料

- Dataset: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
- Reference Project: https://github.com/tajammulbasheer/face-mask-detection
- VGG19 Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"

## 📝 環境需求

- Python 3.10+
- TensorFlow 2.13+
- CUDA 11.8+ (GPU 訓練，可選)
- 8GB+ RAM

## 🏆 成就

✅ 99.75% 驗證準確率
✅ 完整的端到端系統
✅ 互動式 Gradio Demo
✅ 支援多人臉同時偵測

---

*專案建立於 2023-2024 年度深度學習課程*
