"""
Face Mask Detection - Inference Script
對單張圖片進行口罩偵測推理

用法:
    python inference.py --image test.jpg --model results/models/cnn_best_model.h5
"""
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_model(model_path):
    """載入訓練好的模型"""
    print(f"📦 載入模型: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path, target_size=(256, 256)):
    """預處理圖片"""
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取圖片: {image_path}")

    # BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, target_size)

    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch, img


def detect_faces_with_haarcascade(image):
    """使用 Haar Cascade 偵測人臉"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces


def predict_single_image(model, image_path):
    """對單張圖片進行預測"""
    # 預處理
    img_batch, original_img = preprocess_image(image_path)

    # 預測
    prediction = model.predict(img_batch, verbose=0)[0][0]

    # 判斷類別
    if prediction < 0.5:
        label = "WITH MASK"
        confidence = (1 - prediction) * 100
    else:
        label = "NO MASK"
        confidence = prediction * 100

    print(f"🎯 預測結果: {label} (信心度: {confidence:.2f}%)")

    return label, confidence


def predict_with_face_detection(model, image_path, output_path=None):
    """
    使用人臉偵測 + 口罩分類
    對圖片中的所有人臉進行口罩偵測
    """
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取圖片: {image_path}")

    # 偵測人臉
    faces = detect_faces_with_haarcascade(img)
    print(f"👤 偵測到 {len(faces)} 張人臉")

    # 對每張人臉進行分類
    for i, (x, y, w, h) in enumerate(faces):
        # 擷取人臉區域
        face = img[y:y+h, x:x+w]

        # 預處理
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (256, 256))
        face_normalized = face_resized.astype('float32') / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)

        # 預測
        prediction = model.predict(face_batch, verbose=0)[0][0]

        # 判斷類別和顏色
        if prediction < 0.5:
            label = "WITH MASK"
            color = (0, 255, 0)  # Green
            confidence = (1 - prediction) * 100
        else:
            label = "NO MASK"
            color = (0, 0, 255)  # Red
            confidence = prediction * 100

        print(f"  人臉 {i+1}: {label} ({confidence:.1f}%)")

        # 畫框和標籤
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        text = f"{label} {confidence:.1f}%"
        cv2.putText(img, text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 儲存結果
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"💾 結果已儲存至: {output_path}")

    return img, faces


def main():
    parser = argparse.ArgumentParser(description='口罩偵測推理')
    parser.add_argument('--image', type=str, required=True,
                       help='輸入圖片路徑')
    parser.add_argument('--model', type=str,
                       default='results/models/cnn_best_model.h5',
                       help='模型路徑')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出圖片路徑 (可選)')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'face-detection'],
                       help='推理模式: single (整張圖) 或 face-detection (偵測人臉)')

    args = parser.parse_args()

    # 載入模型
    model = load_model(args.model)

    # 推理
    if args.mode == 'single':
        predict_single_image(model, args.image)
    else:
        output_path = args.output or 'results/inference_result.jpg'
        predict_with_face_detection(model, args.image, output_path)

    print("✅ 推理完成！")


if __name__ == '__main__':
    main()
