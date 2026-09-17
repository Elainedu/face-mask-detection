# Face Mask Detection

> Binary face-mask classification with a custom CNN and VGG19 transfer learning, plus optional face-cropping for end-to-end inference.

## Table of Contents

- [What This Does](#what-this-does)
- [Model and Dataset](#model-and-dataset)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Training Pipeline](#training-pipeline)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Running Demo and Inference](#running-demo-and-inference)
- [Training from Scratch](#training-from-scratch)
- [Notebooks](#notebooks)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [License](#license)

## What This Does

Given an input image, the system decides whether the person in it is wearing a
face mask.

- **Input**: an RGB image (any size; resized to 256 x 256 internally).
- **Output**: a binary label `with_mask` / `without_mask` with a sigmoid
  confidence score in `[0, 1]`.
- **End-to-end mode**: `inference.py` and `demo.py` can first run an OpenCV
  Haar-cascade face detector, crop each face, and classify masks per-face for
  multi-person images. Bounding boxes are drawn green for masked faces and
  red for unmasked faces.

Motivation: automatic mask compliance during COVID-19 (entrance control,
public-space monitoring).

## Model and Dataset

Two independent classifiers ship in this repo. They solve the same task
with different backbones:

| Approach | Backbone | Input | Params | Reported val acc |
|----------|----------|-------|--------|------------------|
| Custom CNN | 4 x Conv + Dense (from scratch) | 256 x 256 x 3 | ~1.6 M | 99.75 % |
| Transfer learning | VGG19 (ImageNet pretrained, frozen) + custom head | 256 x 256 x 3 | ~20 M | 99.6 %+ |

Both output a single sigmoid neuron (binary cross-entropy). The label
convention is `0 = without_mask`, `1 = with_mask`.

**Dataset**: [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
from Kaggle.

| Split | Images |
|-------|--------|
| Train | 10,000 |
| Val   | 800 |
| Test  | 992 |

Class distribution is roughly 50/50 balanced.

## System Architecture

```
                    +--------------------+
                    | Input image (any)  |
                    +---------+----------+
                              |
              +---------------+----------------+
              |                                |
              v                                v
   +-------------------+           +--------------------------+
   |  Haar face det.   |           |  Direct classification    |
   |  (OpenCV cascade) |           |  (single face per image)  |
   +---------+---------+           +-------------+------------+
             |                                   |
             | one crop per face                 |
             v                                   |
   +-------------------+                         |
   | Resize 256 x 256  |<------------------------+
   | Normalize / 255   |
   +---------+---------+
             |
             v
   +-------------------+       Sigmoid
   |  CNN or VGG19     |------> p in [0, 1]
   |  (.h5 checkpoint) |
   +---------+---------+
             |
             v
   +-------------------------------+
   | Draw box + label + confidence |
   | green = mask, red = no mask   |
   +-------------------------------+
```

## Repository Layout

```
face-mask-detection/
|-- README.md
|-- requirements.txt
|-- train.py               # Train CNN or VGG19
|-- inference.py           # Single image / face-detection CLI
|-- demo.py                # Gradio web UI
|-- notebooks/
|   |-- cnn-face-mask-binary-classification.ipynb
|   |-- cnn-face-mask-detection-training.ipynb
|   `-- vgg19-transfer-learning-face-mask-detection.ipynb
|-- updated/               # Mirrored copy of the current codebase
|   |-- train.py / inference.py / demo.py / requirements.txt
|   `-- notebooks/
`-- _original/             # Archived original coursework submission
    `-- assignment-01-face-mask-detection/
        |-- F112119106/                      # PDF report + zip + notebook
        `-- face-mask-detection-main/        # Reference repo snapshot
```

- **root-level scripts** = the current, canonical version. Run these.
- **`updated/`** = a mirrored working copy kept in sync with root for
  reproducibility.
- **`_original/`** = the original class assignment (report PDF, submitted
  notebook, upstream reference implementation). Kept for provenance; not
  required to run the project.

## Training Pipeline

```
Kaggle 12K dataset  ->  data/train,val,test (with_mask / without_mask)
        |
        v
ImageDataGenerator (rescale 1/255, shear, zoom, samplewise norm)
        |
        v
build_cnn_model()  OR  build_vgg19_model() (imagenet, freeze base)
        |
        v
model.fit(...) with EarlyStopping(patience=3) + ModelCheckpoint
        |
        v
results/models/{cnn,vgg19}_best_model.h5
        |
        v
inference.py / demo.py load .h5 and classify
```

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Trains CNN or VGG19; CLI flags `--model {cnn,vgg19} --epochs N`. Builds Keras `Sequential` model, wires `ImageDataGenerator` with augmentation, EarlyStopping + ModelCheckpoint, saves `.h5` weights and a training-history plot. |
| `inference.py` | CLI for a single image. Preprocesses to 256 x 256, normalizes to `[0,1]`, loads a `.h5` model, prints label + confidence. `--mode face-detection` first runs Haar cascade and classifies each detected face. |
| `demo.py` | Gradio web UI (defaults to `http://localhost:7860`). Wraps a `FaceMaskDetector` class that combines Haar-cascade face detection with the Keras classifier. `--share` publishes a temporary public URL. |
| `notebooks/cnn-face-mask-binary-classification.ipynb` | End-to-end CNN training walk-through, from data loading to plotting. |
| `notebooks/cnn-face-mask-detection-training.ipynb` | Companion notebook focused on the CNN training loop. |
| `notebooks/vgg19-transfer-learning-face-mask-detection.ipynb` | VGG19 transfer-learning experiment. |

### CNN architecture (from `train.py::build_cnn_model`)

```
Input (256 x 256 x 3)
  Conv2D(16, 3x3, ReLU) -> MaxPool(2x2) -> Dropout(0.2)
  Conv2D(32, 3x3, ReLU) -> MaxPool(2x2) -> Dropout(0.2)
  Conv2D(32, 3x3, ReLU) -> MaxPool(2x2) -> Dropout(0.2)
  Conv2D(64, 3x3, ReLU) -> MaxPool(2x2) -> Dropout(0.2)
  Flatten -> Dense(128, ReLU) -> Dropout(0.2)
  Dense(1, Sigmoid)
```

Optimizer: Adam (lr = 1e-4). Loss: binary cross-entropy. Batch size: 16.

## Requirements

- Python 3.10+
- TensorFlow 2.13+ / Keras 2.13+
- OpenCV 4.8+
- Gradio 4.0+
- NumPy, Pandas, Matplotlib, Pillow, tqdm
- 8 GB+ RAM. CUDA 11.8+ optional for GPU training.

Install:

```bash
pip install -r requirements.txt
```

> **Note on TensorFlow**: this project uses `tensorflow>=2.13` and expects a
> matching NumPy (< 2.0 in most builds). If you hit `AttributeError` around
> `np.object_` or DLL load failures on Windows, create a fresh Python 3.10
> venv and install `tensorflow==2.13.*` first, then the rest. The classifier
> is TF/Keras only; nothing here uses PyTorch or YOLO.

## Running Demo and Inference

### Gradio web demo

```bash
python demo.py
# or with a specific checkpoint
python demo.py --model results/models/vgg19_best_model.h5
# public share link (temporary)
python demo.py --share
```

Opens `http://localhost:7860`: upload an image, get auto face detection,
per-face mask classification, colored boxes (green = mask, red = no mask),
and confidence scores.

### Single-image inference

```bash
# Whole-image classification
python inference.py --image test.jpg --model results/models/cnn_best_model.h5

# Face detection + per-face mask classification
python inference.py --image group.jpg --mode face-detection
```

## Training from Scratch

1. Download the [Face Mask 12K dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
   from Kaggle.
2. Extract into `data/` so the structure is:

   ```
   data/
     train/with_mask/*.jpg
     train/without_mask/*.jpg
     val/with_mask/*.jpg
     val/without_mask/*.jpg
     test/with_mask/*.jpg
     test/without_mask/*.jpg
   ```

3. Train:

   ```bash
   # Custom CNN
   python train.py --model cnn --epochs 32

   # VGG19 transfer learning
   python train.py --model vgg19 --epochs 20
   ```

4. Outputs land in `results/`:

   ```
   results/
     models/
       cnn_best_model.h5
       vgg19_best_model.h5
     plots/
       cnn_training_history.png
       vgg19_training_history.png
   ```

Typical CNN run: ~14 min per epoch on CPU, converges around epoch 20; early
stopping (patience = 3) usually cuts it off by epoch 23.

## Notebooks

The `notebooks/` folder holds the original exploration:

- `cnn-face-mask-binary-classification.ipynb` - basic CNN on the mask dataset.
- `cnn-face-mask-detection-training.ipynb` - deeper training loop, plots,
  and evaluation metrics.
- `vgg19-transfer-learning-face-mask-detection.ipynb` - swaps the backbone
  for ImageNet-pretrained VGG19 with a custom classification head.

These are what the CLI scripts are derived from; keep them as reference or
for interactive experimentation.

## Notes

- **GPU is optional.** The CNN trains happily on CPU in a few hours; VGG19
  benefits noticeably from a GPU because of the deeper backbone.
- **Data augmentation matters.** Samplewise normalization made the model
  robust to lighting variation; shear + zoom help with head-pose variety.
- **Custom CNN is enough for this task.** VGG19 converges faster but does
  not out-perform the from-scratch CNN by a meaningful margin on this
  dataset.
- **Webcam / real-time**: not built in. Adapting `inference.py` to a
  `cv2.VideoCapture` loop is straightforward if needed.
- **Face detector**: OpenCV Haar cascade is used purely to crop candidate
  faces. It is fast but coarse - misses profile faces and heavy occlusion.
  Swap in RetinaFace / MTCNN for stricter recall.

## Files Not in Repo

The following are intentionally excluded and must be produced or downloaded
locally:

- `results/models/*.h5` - trained checkpoints (tens to hundreds of MB).
  Retrain with `train.py` to regenerate.
- `data/` - the Kaggle 12K dataset. Download from the link above.
- `results/plots/*.png` - training curves; regenerated on each run.

## License

Released for educational use as part of a deep learning course project.
The VGG19 pretrained weights (ImageNet) and the Kaggle Face Mask 12K
dataset remain under their original licenses. Please respect them if you
redistribute derived work.
