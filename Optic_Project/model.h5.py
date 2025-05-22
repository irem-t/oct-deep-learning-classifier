import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# --------------------------
# 1. Veri Yükleme Fonksiyonu
# --------------------------
def load_dataset(data_dir, image_size=(128, 128)):
    images = []
    labels = []
    class_names = ["NORMAL", "DR", "DRUSEN", "MH", "AMD", "CNV", "CSR", "DME"]

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"[!] Uyarı: '{class_path}' dizini bulunamadı.")
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)

    X = np.array(images).reshape(-1, image_size[0], image_size[1], 1)
    y = np.array(labels)
    return X, y, class_names

# --------------------------
# 2. Model Oluşturma Fonksiyonu
# --------------------------
def build_model(input_shape=(128, 128, 1), num_classes=8):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')  # Çoklu sınıf tahmini
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --------------------------
# 3. Ana Fonksiyon
# --------------------------
def main():
    data_dir = r"C:\Optic_Project\RetinalOCT_Dataset\RetinalOCT_Dataset\test"
    image_size = (128, 128)

    print("[📂] Veriler yükleniyor...")
    X, y, class_names = load_dataset(data_dir, image_size)

    print("Veri şekilleri:", X.shape, y.shape)
    print("Sınıflar:", class_names)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[✔] {len(X_train)} eğitim, {len(X_val)} doğrulama örneği")

    model = build_model(input_shape=(128, 128, 1), num_classes=len(class_names))

    checkpoint = ModelCheckpoint("best_model.h5",
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max')

    print("[🧠] Model eğitiliyor...")
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint])

    print("\n[🔍] Model değerlendiriliyor...")
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nFinal Validation Metrics:")
    print(f"Accuracy: {val_acc:.4f}")

    model.save("final_model.h5")
    print("[💾] Model 'final_model.h5' olarak kaydedildi.")

    best_model = load_model("best_model.h5")
    print("[🏆] En iyi model yüklendi")

if __name__ == "__main__":
    main()
