"""
===============================================================================
專案名稱: Face Mask Detection System - 口罩偵測系統
===============================================================================

[專案簡介]
這是一個使用深度學習技術的口罩配戴偵測系統，可以自動辨識圖片中的人臉，
並判斷每個人是否有正確配戴口罩。系統結合了傳統電腦視覺（人臉偵測）和
深度學習（口罩分類）技術，達到高準確率的即時偵測。

[技術架構]
- 人臉偵測: OpenCV Haar Cascade Classifier
- 分類模型: Custom CNN / VGG19 Transfer Learning
- 深度學習框架: TensorFlow/Keras
- Web 介面: Gradio
- 資料增強: ImageDataGenerator (旋轉、縮放、翻轉、亮度調整)

[訓練資料]
- 資料集: Kaggle Face Mask 12K Dataset
- 類別: WITH MASK (有戴口罩) / WITHOUT MASK (沒戴口罩)
- 訓練集: ~10,000 張圖片
- 驗證集: ~2,000 張圖片

[模型效能]
- CNN 模型準確率: ~99.75%
- VGG19 模型準確率: ~99.50%
- 推論速度: < 1 秒/張 (CPU)
- 支援多人臉同時偵測

[主要功能]
1. 自動人臉偵測 - 使用 Haar Cascade 快速定位人臉位置
2. 口罩配戴分類 - 深度學習模型判斷是否配戴口罩
3. 視覺化標註 - 在圖片上繪製偵測框和分類結果
4. 信心度顯示 - 顯示模型預測的信心程度 (0-100%)
5. 統計資訊 - 計算配戴/未配戴口罩的人數

[啟動方式]
基本啟動（使用預設 CNN 模型）:
    python demo.py

使用 VGG19 模型:
    python demo.py --model results/models/vgg19_best_model.h5

產生公開分享連結（可讓其他人透過網路訪問）:
    python demo.py --share

指定 Port:
    python demo.py --port 8080

[使用說明]
1. 執行程式後，會在瀏覽器自動開啟 http://127.0.0.1:7860
2. 點擊上傳區域選擇圖片（支援 JPG, PNG 等格式）
3. 系統會自動進行以下處理：
   - 偵測圖片中所有人臉
   - 對每張臉判斷是否配戴口罩
   - 繪製綠色框（有戴口罩）或紅色框（沒戴口罩）
   - 顯示每張臉的分類結果和信心度
4. 查看右側的統計資訊

[面試展示重點]
1. **技術深度**: 說明 VGG19 Transfer Learning 的原理和優勢
2. **資料增強**: 解釋如何使用 ImageDataGenerator 增加訓練資料多樣性
3. **預處理一致性**: 強調訓練和推論時的預處理必須一致（samplewise normalization）
4. **實際應用**: 討論在疫情期間的實際應用場景（公共場所監控、門禁系統）
5. **改進方向**: 可提及即時影片串流偵測、口罩配戴正確性判斷等進階功能

[檔案結構]
demo.py                          # 本檔案 - Gradio Web 展示介面
train.py                         # 模型訓練程式
train_vgg19.py                   # VGG19 遷移學習訓練程式
results/models/                  # 訓練好的模型檔案
    ├── cnn_best_model.h5       # 自訂 CNN 模型
    └── vgg19_best_model.h5     # VGG19 模型
data/                            # 資料集目錄（需自行下載）
    ├── train/
    └── validation/

[注意事項]
- 首次執行前需先訓練模型: python train.py
- 人臉偵測在正面、光線充足的情況下效果最佳
- 支援單張圖片多人臉偵測
- 預處理時必須使用 samplewise normalization 以確保與訓練時一致

[開發者]
碩士班課程專案 - 深度學習
建立日期: 2024
更新日期: 2026-03-10 (修正 Gradio 6.0 相容性和預處理一致性)

===============================================================================
"""
# ==================== 匯入必要套件 ====================
import argparse  # 處理命令列參數
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隱藏 TensorFlow 警告訊息

import cv2  # OpenCV，用於影像處理和人臉偵測
import numpy as np  # 數值運算
import gradio as gr  # 建立 Web 介面的工具
import tensorflow as tf  # 深度學習框架
from tensorflow import keras  # 載入訓練好的模型
from PIL import Image  # 處理圖片格式


class FaceMaskDetector:
    """
    口罩偵測器類別（包含所有偵測和分類的功能）

    這個類別會做兩件事：
    1. 偵測圖片中的人臉位置（用 OpenCV）
    2. 判斷每張臉是否有戴口罩（用訓練好的深度學習模型）
    """

    def __init__(self, model_path):
        """
        初始化偵測器（準備好模型和人臉偵測器）

        參數:
            model_path: 訓練好的模型檔案路徑 (.h5 檔案)
        """
        print(f"[LOAD] 載入模型: {model_path}")

        # ===== 1. 載入訓練好的口罩分類模型 =====
        # 這個模型可以判斷一張人臉圖片是否有戴口罩
        self.model = keras.models.load_model(model_path)

        # ===== 2. 載入人臉偵測器 =====
        # 使用 OpenCV 內建的 Haar Cascade 人臉偵測器
        # 這是一個傳統的機器學習方法，可以快速找到圖片中的人臉位置
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def preprocess_face(self, face_img):
        """
        預處理人臉圖片（讓它符合模型的輸入格式）

        模型訓練時用的是 256x256 的 RGB 圖片，像素值在 0~1 之間
        所以這裡要把偵測到的人臉調整成一樣的格式

        參數:
            face_img: 從原圖裁切出來的人臉圖片 (BGR 格式)

        回傳:
            處理好的人臉圖片 (shape: (1, 256, 256, 3))
        """
        # ===== 步驟 1: BGR 轉 RGB =====
        # OpenCV 預設是 BGR，但模型訓練時用 RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # ===== 步驟 2: 調整大小為 256x256 =====
        face_resized = cv2.resize(face_rgb, (256, 256))

        # ===== 步驟 3: 正規化像素值到 0~1 =====
        # 原本是 0~255，除以 255 變成 0~1
        face_normalized = face_resized.astype('float32') / 255.0

        # ===== 步驟 4: Samplewise 中心化和標準化 =====
        # 訓練時使用了 samplewise_center 和 samplewise_std_normalization
        # 所以預測時也要做一樣的處理
        mean = np.mean(face_normalized)  # 計算平均值
        std = np.std(face_normalized)    # 計算標準差
        if std > 0:  # 避免除以 0
            face_normalized = (face_normalized - mean) / std  # 中心化並標準化
        else:
            face_normalized = face_normalized - mean  # 只做中心化

        # ===== 步驟 5: 增加 batch 維度 =====
        # 模型期望輸入是 (batch_size, height, width, channels)
        # 我們只有一張圖，所以 batch_size=1
        face_batch = np.expand_dims(face_normalized, axis=0)

        return face_batch

    def detect_and_classify(self, image):
        """
        偵測人臉並分類是否戴口罩（這是整個系統的核心功能）

        流程:
        1. 偵測圖片中所有的人臉位置
        2. 對每張人臉判斷是否有戴口罩
        3. 在圖片上畫框並標註結果
        4. 回傳標註後的圖片和文字說明

        參數:
            image: 使用者上傳的圖片 (PIL Image 或 numpy array)

        回傳:
            result_image: 標註後的圖片（畫了框和標籤）
            summary: 文字說明（偵測結果統計）
        """
        # ===== 步驟 1: 統一圖片格式 =====
        # Gradio 可能傳入 PIL Image 或 numpy array，統一轉成 numpy array
        if isinstance(image, Image.Image):
            img = np.array(image)  # PIL Image 轉 numpy
        else:
            img = image  # 已經是 numpy array

        # ===== 步驟 2: 轉換顏色格式給 OpenCV 使用 =====
        # Gradio 傳入的是 RGB，OpenCV 需要 BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ===== 步驟 3: 偵測人臉位置 =====
        # Haar Cascade 需要灰階圖片
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 執行人臉偵測，回傳所有人臉的座標 (x, y, w, h)
        faces = self.face_cascade.detectMultiScale(
            gray,  # 灰階圖片
            scaleFactor=1.1,  # 圖片縮放比例（用於偵測不同大小的臉）
            minNeighbors=5,  # 偵測品質（越高越嚴格，誤判越少）
            minSize=(30, 30)  # 最小人臉尺寸（太小的不偵測）
        )

        # ===== 步驟 4: 準備輸出資料 =====
        result_img = img.copy()  # 複製原圖，待會要在上面畫框
        summary_lines = [f"🔍 偵測到 {len(faces)} 張人臉\n"]  # 文字說明

        # 如果沒偵測到人臉，直接回傳
        if len(faces) == 0:
            summary_lines.append("❌ 未偵測到人臉，請上傳包含人臉的圖片")
            return result_img, "\n".join(summary_lines)

        # ===== 步驟 5: 對每張人臉進行口罩分類 =====
        mask_count = 0  # 戴口罩的人數
        no_mask_count = 0  # 沒戴口罩的人數

        for i, (x, y, w, h) in enumerate(faces):
            # ----- 5.1: 擷取人臉區域 -----
            face = img_bgr[y:y+h, x:x+w]  # 從原圖裁切出人臉

            # ----- 5.2: 用模型預測是否戴口罩 -----
            face_batch = self.preprocess_face(face)  # 預處理成模型需要的格式
            prediction = self.model.predict(face_batch, verbose=0)[0][0]  # 預測，得到 0~1 的數值

            # ----- 5.3: 判斷類別 -----
            # 模型訓練時：0 = WITH MASK, 1 = WITHOUT MASK
            # 所以 prediction < 0.5 表示有戴口罩
            if prediction < 0.5:
                label = "WITH MASK"  # 有戴口罩
                color = (0, 255, 0)  # 綠色框
                confidence = (1 - prediction) * 100  # 信心度（越接近 0 越確定）
                mask_count += 1
            else:
                label = "NO MASK"  # 沒戴口罩
                color = (255, 0, 0)  # 紅色框
                confidence = prediction * 100  # 信心度（越接近 1 越確定）
                no_mask_count += 1

            # ----- 5.4: 在圖片上畫框 -----
            # 注意：result_img 是 RGB 格式，但 color 是 BGR，要轉換
            cv2.rectangle(result_img, (x, y), (x+w, y+h),
                         color[::-1], 3)  # BGR to RGB，線條粗細 3

            # ----- 5.5: 畫標籤背景 -----
            # 先算出標籤文字的大小
            text = f"{label}"
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            # 畫一個實心矩形當背景（讓文字更清楚）
            cv2.rectangle(result_img, (x, y-text_h-10), (x+text_w, y),
                         color[::-1], -1)  # -1 表示填滿

            # ----- 5.6: 寫上標籤文字 -----
            cv2.putText(result_img, text, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)  # 白色文字

            # ----- 5.7: 顯示信心度 -----
            conf_text = f"{confidence:.1f}%"
            cv2.putText(result_img, conf_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       color[::-1], 2)

            # ----- 5.8: 記錄結果到文字說明 -----
            emoji = "✅" if label == "WITH MASK" else "❌"
            summary_lines.append(
                f"{emoji} 人臉 {i+1}: {label} (信心度: {confidence:.1f}%)"
            )

        # ===== 步驟 6: 加入統計摘要 =====
        summary_lines.append(f"\n📊 統計:")
        summary_lines.append(f"   戴口罩: {mask_count} 人")
        summary_lines.append(f"   未戴口罩: {no_mask_count} 人")

        # ===== 步驟 7: 回傳結果 =====
        return result_img, "\n".join(summary_lines)


def create_demo(model_path):
    """
    建立 Gradio 網頁介面（讓使用者可以透過瀏覽器上傳圖片）

    Gradio 會自動產生一個美觀的網頁，包含：
    - 圖片上傳區
    - 偵測結果顯示區
    - 說明文字

    參數:
        model_path: 訓練好的模型路徑

    回傳:
        demo: Gradio 介面物件
    """
    # ===== 步驟 1: 建立偵測器實例 =====
    detector = FaceMaskDetector(model_path)

    # ===== 步驟 2: 準備範例圖片（選用）=====
    # 如果有範例圖片，可以放在這裡讓使用者快速測試
    examples = [
        # 範例: ["data/examples/with_mask.jpg"]
        # 範例: ["data/examples/without_mask.jpg"]
    ]

    # ===== 步驟 3: 建立 Gradio 介面 =====
    demo = gr.Interface(
        # --- 核心功能 ---
        fn=detector.detect_and_classify,

        # --- 輸入元件 ---
        inputs=gr.Image(type="pil", label="上傳圖片"),

        # --- 輸出元件 ---
        outputs=[
            gr.Image(type="numpy", label="偵測結果"),
            gr.Textbox(label="分析結果", lines=10)
        ],

        # --- 介面外觀 ---
        title="Face Mask Detection Demo",
        description="""
        ### 口罩偵測系統
        上傳包含人臉的圖片，系統會自動偵測並判斷是否配戴口罩。

        **功能特色:**
        - 自動人臉偵測
        - 口罩配戴判斷
        - 多人臉同時分析
        - 信心度顯示

        **使用說明:**
        1. 上傳圖片（支援 JPG, PNG）
        2. 等待系統分析
        3. 查看標註結果和統計資訊

        **訓練資料:** Kaggle Face Mask 12K Dataset
        **模型架構:** Custom CNN / VGG19
        **準確率:** ~99.75%
        """,

        # --- CSS 樣式 ---
        css="""
        textarea {
            font-size: 1.2rem !important;
            line-height: 1.6 !important;
        }
        """,

        # --- 其他設定 ---
        examples=examples if examples else None
    )

    return demo

def main():
    """
    主程式進入點（啟動 Gradio 網頁介面）

    使用方式:
        python demo.py                                         # 使用預設模型
        python demo.py --model results/models/vgg19_best_model.h5  # 指定模型
        python demo.py --share                                 # 產生公開連結
        python demo.py --port 8080                            # 指定 port
    """
    # ===== 步驟 1: 設定命令列參數 =====
    parser = argparse.ArgumentParser(description='啟動口罩偵測 Gradio Demo')

    # 參數 1: 模型路徑
    parser.add_argument('--model', type=str,
                       default='results/models/cnn_best_model.h5',  # 預設使用 CNN 模型
                       help='模型路徑（.h5 檔案）')

    # 參數 2: 是否產生公開分享連結
    parser.add_argument('--share', action='store_true',
                       help='建立公開分享連結（可以讓其他人透過網路訪問）')

    # 參數 3: 網頁伺服器的 port
    parser.add_argument('--port', type=int, default=7860,
                       help='Port 號（預設 7860）')

    args = parser.parse_args()  # 解析參數

    # ===== 步驟 2: 檢查模型檔案是否存在 =====
    # 如果沒有訓練過模型，模型檔案不存在，要提醒使用者先訓練
    if not os.path.exists(args.model):
        print(f"[ERROR] 找不到模型: {args.model}")
        print(f"")
        print(f"請先執行訓練: python train.py")
        print(f"或指定其他模型: python demo.py --model <模型路徑>")
        return  # 結束程式

    # ===== 步驟 3: 建立 Gradio 介面 =====
    print(f"[START] 啟動 Gradio Demo...")
    print(f"[INFO] 使用模型: {args.model}")
    demo = create_demo(args.model)

    # ===== 步驟 4: 啟動網頁伺服器 =====
    demo.launch(
    share=args.share,
    server_port=args.port,
    css="textarea { font-size: 1.5rem !important; }"
)

# ===== 程式進入點 =====
# 當執行 "python demo.py" 時，Python 會從這裡開始執行
if __name__ == '__main__':
    main()
