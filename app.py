import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps, ImageGrab
import numpy as np
import tensorflow as tf
import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
MODEL_PATH = "mnist_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. Train your model first using train.py."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Create the main application window
window = Tk()
window.title("Handwritten Digit Recognizer")
window.geometry("400x480")
window.configure(bg="#ECECEC")

# Canvas to draw digits
canvas_width = 280
canvas_height = 280
canvas = Canvas(window, width=canvas_width, height=canvas_height, bg='white', cursor="cross")
canvas.pack(pady=20)

# Create a drawing object
image1 = Image.new("RGB", (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image1)

# Functions
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_width, canvas_height), fill='white')
    result_label.config(text="")

def draw_lines(event):
    x, y = event.x, event.y
    r = 8  # brush size
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

def predict_digit():
    # Grab canvas content
    canvas.update()
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab(bbox=(x, y, x1, y1))
    img = img.convert("L")  # grayscale
    img = ImageOps.invert(img)  # black background -> white digit

    # Binarize image
    img = img.point(lambda p: 255 if p > 100 else 0, '1')

    # Crop to digit area
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Make image square (padding)
    width, height = img.size
    new_size = max(width, height)
    new_img = Image.new('L', (new_size, new_size), 0)
    new_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
    img = new_img.resize((28, 28))

    # Normalize and reshape
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(img)
    digit = np.argmax(pred)
    confidence = np.max(pred)

    result_label.config(
        text=f"Predicted Digit: {digit}\nConfidence: {confidence*100:.2f}%",
        font=("Helvetica", 16, "bold")
    )



# UI Elements
button_frame = Frame(window, bg="#ECECEC")
button_frame.pack(pady=10)

clear_button = Button(button_frame, text="Clear", command=clear_canvas, bg="#f55", fg="white", width=10)
clear_button.grid(row=0, column=0, padx=10)

predict_button = Button(button_frame, text="Predict", command=predict_digit, bg="#4CAF50", fg="white", width=10)
predict_button.grid(row=0, column=1, padx=10)

result_label = Label(window, text="", bg="#ECECEC", font=("Helvetica", 14))
result_label.pack(pady=20)

canvas.bind("<B1-Motion>", draw_lines)

# Start the main loop
window.mainloop()

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Define prediction function
def predict_digit(image):
    # image is a PIL Image from Gradio
    img = image.convert("L")                 # grayscale
    img = ImageOps.invert(img)               # invert: make digit white on black
    img = img.resize((28, 28))               # MNIST size
    arr = np.array(img).astype("float32")/255.0
    arr = arr.reshape(1, 28, 28, 1)
    pred = model.predict(arr, verbose=0)[0]
    # Return a label mapping {class: probability}
    return {str(i): float(pred[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Draw or Upload a digit"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Handwritten Digit Recognizer",
    description="Draw a single digit (0–9) or upload an image. Model trained on MNIST.",
    allow_flagging="never",
    live=False
)

# Launch the app
if __name__ == "__main__":
    demo.launch()