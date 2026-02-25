"""
Face Mask Detection - Training Script
使用 CNN 或 VGG19 訓練口罩偵測模型

用法:
    python train.py --model cnn --epochs 32
    python train.py --model vgg19 --epochs 20
"""
# ==================== 匯入必要套件 ====================
import argparse  # 用於處理命令列參數
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隱藏 TensorFlow 的警告訊息

import tensorflow as tf  # 深度學習框架
from tensorflow import keras
from tensorflow.keras import layers  # 神經網路層
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 圖片資料增強工具
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # 訓練過程的回調函數
import matplotlib.pyplot as plt  # 繪圖工具


def build_cnn_model(input_shape=(256, 256, 3)):
    """
    建立自定義 CNN 模型（卷積神經網路）

    這個模型會學習辨識人臉圖片中是否有戴口罩
    輸入: 256x256 的彩色圖片 (3個顏色通道: RGB)
    輸出: 0~1 之間的數值 (0=沒戴口罩, 1=有戴口罩)
    """
    model = keras.Sequential([
        # ===== Block 1: 第一層卷積層 =====
        # 用 16 個 3x3 的濾鏡掃描圖片，找出基本特徵（如邊緣、線條）
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),  # 將圖片縮小一半，減少運算量
        layers.Dropout(0.2),  # 隨機丟棄 20% 的神經元，防止過擬合

        # ===== Block 2: 第二層卷積層 =====
        # 用 32 個濾鏡找出更複雜的特徵（如形狀、紋理）
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),  # 再次縮小圖片
        layers.Dropout(0.2),

        # ===== Block 3: 第三層卷積層 =====
        # 繼續用 32 個濾鏡找出更抽象的特徵
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),

        # ===== Block 4: 第四層卷積層 =====
        # 用 64 個濾鏡找出高階特徵（如口罩的形狀、位置）
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),

        # ===== 全連接層：整合特徵並做出判斷 =====
        layers.Flatten(),  # 將 2D 特徵圖攤平成 1D 向量
        layers.Dense(128, activation='relu'),  # 128 個神經元整合所有特徵
        layers.Dropout(0.2),  # 防止過擬合
        layers.Dense(1, activation='sigmoid')  # 最後一層輸出 0~1 的機率值
    ])

    return model


def build_vgg19_model(input_shape=(256, 256, 3)):
    """
    建立 VGG19 遷移學習模型（使用預訓練的模型）

    VGG19 是一個在 ImageNet 資料集上訓練好的模型
    我們使用「遷移學習」：保留 VGG19 已學會的特徵提取能力
    只訓練最後的分類層，這樣訓練速度快且準確率高
    """
    # ===== 載入 VGG19 預訓練模型 =====
    base_model = keras.applications.VGG19(
        include_top=False,  # 不要最後的分類層（我們要自己加）
        weights='imagenet',  # 使用在 ImageNet 上訓練好的權重
        input_shape=input_shape  # 輸入圖片大小
    )
    base_model.trainable = False  # 凍結 VGG19 的權重，不訓練它（節省時間）

    # ===== 在 VGG19 上面加上我們自己的分類層 =====
    model = keras.Sequential([
        base_model,  # VGG19 負責提取特徵
        layers.Flatten(),  # 攤平特徵
        layers.Dense(256, activation='relu'),  # 256 個神經元整合特徵
        layers.Dropout(0.5),  # 丟棄 50% 神經元防止過擬合
        layers.Dense(1, activation='sigmoid')  # 輸出 0~1 的機率（戴口罩/沒戴口罩）
    ])

    return model


def prepare_data(data_dir='data', batch_size=16):
    """
    準備訓練和驗證資料（載入圖片並進行資料增強）

    資料集結構:
    data/
      train/           # 訓練資料
        with_mask/     # 有戴口罩的圖片
        without_mask/  # 沒戴口罩的圖片
      val/             # 驗證資料（用來評估模型表現）
        with_mask/
        without_mask/
    """
    # ===== 訓練資料的前處理與增強 =====
    # 資料增強可以讓模型看到更多變化的圖片，提升泛化能力
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # 將像素值從 0-255 縮放到 0-1（神經網路喜歡小數值）
        shear_range=0.2,  # 隨機剪切變形（讓模型學會辨識各種角度的臉）
        zoom_range=0.2,  # 隨機放大縮小（讓模型學會辨識遠近不同的臉）
        horizontal_flip=True,  # 隨機水平翻轉（增加資料多樣性）
        samplewise_center=True,  # 將每張圖片的像素值中心化（減去平均值）
        samplewise_std_normalization=True  # 標準化（除以標準差）
    )

    # ===== 驗證資料的前處理 =====
    # 驗證資料不需要增強，只需要正規化
    val_datagen = ImageDataGenerator(
        rescale=1./255,  # 縮放像素值
        samplewise_center=True,  # 中心化
        samplewise_std_normalization=True  # 標準化
    )

    # ===== 從資料夾載入圖片 =====
    # 自動根據資料夾名稱（with_mask/without_mask）標記類別
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),  # 訓練資料路徑
        target_size=(256, 256),  # 將所有圖片調整為 256x256
        batch_size=batch_size,  # 每次訓練使用幾張圖片
        class_mode='binary'  # 二元分類（0 或 1）
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),  # 驗證資料路徑
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator


def train(model_type='cnn', epochs=32, batch_size=16, data_dir='data'):
    """
    訓練模型的主函數（這是整個訓練流程的核心）

    參數:
        model_type: 'cnn' 或 'vgg19'（選擇要用哪種模型）
        epochs: 訓練幾輪（每一輪都會看過所有訓練資料一次）
        batch_size: 每次訓練用幾張圖片
        data_dir: 資料集路徑
    """
    print(f"🚀 開始訓練 {model_type.upper()} 模型...")

    # ===== 步驟 1: 建立模型架構 =====
    if model_type == 'cnn':
        model = build_cnn_model()  # 使用自訂 CNN
    elif model_type == 'vgg19':
        model = build_vgg19_model()  # 使用 VGG19 遷移學習
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ===== 步驟 2: 編譯模型（設定訓練方式）=====
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # 使用 Adam 優化器，學習率 0.0001
        loss='binary_crossentropy',  # 二元分類的損失函數
        metrics=['accuracy']  # 追蹤準確率
    )

    print(f"📊 模型參數: {model.count_params():,}")  # 顯示模型有多少個參數需要訓練

    # ===== 步驟 3: 載入訓練資料 =====
    train_gen, val_gen = prepare_data(data_dir, batch_size)

    # ===== 步驟 4: 設定訓練過程的回調函數 =====
    callbacks = [
        # EarlyStopping: 如果驗證集的 loss 連續 3 輪都沒改善，就停止訓練（避免浪費時間）
        EarlyStopping(
            monitor='val_loss',  # 監控驗證集的 loss
            patience=3,  # 等待 3 輪沒改善才停止
            restore_best_weights=True,  # 停止時恢復到最好的權重
            verbose=1  # 顯示停止訊息
        ),
        # ModelCheckpoint: 每當驗證準確率創新高，就自動儲存模型
        ModelCheckpoint(
            f'results/models/{model_type}_best_model.h5',  # 儲存路徑
            monitor='val_accuracy',  # 監控驗證準確率
            save_best_only=True,  # 只儲存最好的模型
            verbose=1  # 顯示儲存訊息
        )
    ]

    # ===== 步驟 5: 開始訓練！=====
    # 這裡會跑很久，模型會不斷調整參數來提高準確率
    history = model.fit(
        train_gen,  # 訓練資料
        epochs=epochs,  # 訓練幾輪
        validation_data=val_gen,  # 驗證資料（用來評估模型是否過擬合）
        callbacks=callbacks,  # 使用上面設定的回調函數
        verbose=1  # 顯示訓練進度
    )

    # ===== 步驟 6: 儲存最終模型 =====
    model.save(f'results/models/{model_type}_final_model.h5')
    print(f"✅ 模型已儲存至 results/models/{model_type}_final_model.h5")

    # ===== 步驟 7: 繪製訓練過程的曲線圖 =====
    plot_history(history, model_type)

    return model, history  # 回傳訓練好的模型和訓練歷史


def plot_history(history, model_type):
    """
    繪製訓練曲線（視覺化訓練過程）

    可以從圖表中看出：
    - 模型是否有學到東西（loss 是否下降、accuracy 是否上升）
    - 是否過擬合（train 和 val 的曲線是否差很多）
    """
    # ===== 建立一個有兩個子圖的畫布 =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1行2列的圖表

    # ===== 左圖: Loss（損失）曲線 =====
    # Loss 越小表示模型預測越準確
    ax1.plot(history.history['loss'], label='Train Loss')  # 訓練集的 loss
    ax1.plot(history.history['val_loss'], label='Val Loss')  # 驗證集的 loss
    ax1.set_xlabel('Epoch')  # X 軸：訓練輪數
    ax1.set_ylabel('Loss')  # Y 軸：損失值
    ax1.set_title('Training and Validation Loss')
    ax1.legend()  # 顯示圖例
    ax1.grid(True)  # 顯示格線

    # ===== 右圖: Accuracy（準確率）曲線 =====
    # Accuracy 越高表示分類越準確
    ax2.plot(history.history['accuracy'], label='Train Acc')  # 訓練集準確率
    ax2.plot(history.history['val_accuracy'], label='Val Acc')  # 驗證集準確率
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # ===== 儲存圖表 =====
    plt.tight_layout()  # 自動調整子圖間距
    plt.savefig(f'results/plots/{model_type}_training_history.png', dpi=150)  # 儲存為圖片
    print(f"📈 訓練曲線已儲存至 results/plots/{model_type}_training_history.png")


def main():
    """
    主程式進入點（從命令列執行時會呼叫這個函數）

    使用方式:
        python train.py                              # 使用預設參數
        python train.py --model cnn --epochs 32      # 訓練 CNN 模型 32 輪
        python train.py --model vgg19 --epochs 20    # 訓練 VGG19 模型 20 輪
    """
    # ===== 設定命令列參數 =====
    parser = argparse.ArgumentParser(description='訓練口罩偵測模型')

    # 參數 1: 選擇模型類型
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'vgg19'],  # 只能選這兩種
                       help='模型類型: cnn 或 vgg19')

    # 參數 2: 訓練輪數
    parser.add_argument('--epochs', type=int, default=32,
                       help='訓練 epochs 數量（完整看過所有資料幾次）')

    # 參數 3: 批次大小
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size（每次訓練用幾張圖片）')

    # 參數 4: 資料集路徑
    parser.add_argument('--data-dir', type=str, default='data',
                       help='資料集路徑')

    args = parser.parse_args()  # 解析命令列參數

    # ===== 建立結果資料夾 =====
    # exist_ok=True 表示資料夾存在也不會報錯
    os.makedirs('results/models', exist_ok=True)  # 存放訓練好的模型
    os.makedirs('results/plots', exist_ok=True)   # 存放訓練曲線圖

    # ===== 開始訓練 =====
    train(
        model_type=args.model,      # 使用哪種模型
        epochs=args.epochs,          # 訓練幾輪
        batch_size=args.batch_size,  # 批次大小
        data_dir=args.data_dir       # 資料集路徑
    )

    print("🎉 訓練完成！")


# ===== 程式進入點 =====
# 當你執行 "python train.py" 時，Python 會從這裡開始執行
if __name__ == '__main__':
    main()
