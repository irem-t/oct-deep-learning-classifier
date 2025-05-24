import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import tensorflow as tf

# Class names
class_names = ["NORMAL", "DR", "DRUSEN", "MH", "AMD", "CNV", "CSR", "DME"]

# Turn image to the appropriate format
def preprocess_image(path, image_size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image could not be analyzed!")
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, 128, 128, 1)
    return img

# Loading the model
def load_trained_model(model_path):
    model = load_model(model_path, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    print("[✓] The model is loaded and compiled.")
    return model




# Prediction function to be called from the GUI (no parameters)
def predict_image():
    global model, image_label, photo_img
    file_path = filedialog.askopenfilename(
        title="Select and OCT image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("Image could not be analyzed!")
        return

    try:
        # Prediction operations
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100
        result_text = f"🧠 Prediction: {class_names[predicted_class]} ({confidence:.2f}%)"
        print(result_text)
        result_label.config(text=result_text)

        # Display the image (in color)
        img_color = cv2.imread(file_path)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        # Convert image to PIL format
        img_pil = Image.fromarray(img_color)
        img_pil = img_pil.resize((300, 300))  # Display size in the GUI
        photo_img = ImageTk.PhotoImage(img_pil)
        image_label.config(image=photo_img)

    except Exception as e:
        result_label.config(text=f"Error: {e}")
        print(f"Error: {e}")










# === GUI is starting ===
model = load_trained_model("best_model.h5")

window = tk.Tk()
window.title("Clasification of OCT image")
window.geometry("650x550")
window.resizable(False, False)

title = tk.Label(window, text="Classification of OCT image", font=("Arial", 14, "bold"))
title.pack(pady=10)

btn_image = tk.Button(window, text="Select an Image and Predict", width=25, command=predict_image)
btn_image.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 12), fg="magenta")
result_label.pack(pady=10)

image_label = tk.Label(window)  # The image will be displayed here
image_label.pack(pady=10)

info_label = tk.Label(
    window,
    text="Supported Classes:\n" + ", ".join(class_names),
    wraplength=500,
    justify="center"
)
info_label.pack(pady=10)

window.mainloop()





