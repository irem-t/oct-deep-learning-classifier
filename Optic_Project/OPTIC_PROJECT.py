import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import tensorflow as tf  # Metrics için

class_names = ["NORMAL", "DR", "DRUSEN", "MH", "AMD", "CNV", "CSR", "DME"]

def preprocess_image(path, image_size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Görsel okunamadı!")
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, 128, 128, 1)
    return img

def load_trained_model(model_path):
    model = load_model(model_path, compile=False)  # Derlemeden yükle
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Çok sınıflı sınıflandırma için
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall()]
    )
    print("[✓] Model yüklendi ve derlendi")
    return model

def predict_image(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    print(f"🧠 Tahmin: {class_names[predicted_class]} (Güven: {confidence:.4f})")

    
    img_to_show = cv2.imread(image_path)
    if img_to_show is not None:
        # Yazı olarak tahmin ve güveni görüntüye ekle
        text = f"{class_names[predicted_class]} ({confidence:.2f})"
        cv2.putText(img_to_show, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Seçilen Görüntü ve Tahmin", img_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
       


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="OCT Görüntüsü Seçin",
        filetypes=[("JPEG", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("Görsel seçilmedi!")
        return

    model = load_trained_model("best_model.h5")
    predict_image(model, file_path)

if __name__ == "__main__":
    main()
