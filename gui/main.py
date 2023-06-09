import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import tkinter as tk
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter,ImageDraw
from tkinter import ttk

root = tk.Tk()
root.geometry("1200x800")
root.title("MRI SEGMANTATION V1.2")
root.config(bg="white")

pen_color = "black"
pen_size = 5
file_path = ""
mode = "draw"
pan_x = 0
pan_y = 0
zoom_scale = 1.0
original_width = 0
original_height = 0

original_image = None


ovals = []


def add_image():
    global file_path, original_width, original_height, original_image
   
    file_path = filedialog.askopenfilename(initialdir="D:/codefirst.io/Tkinter Image Editor/Pictures")
    image = Image.open(file_path)
    original_width, original_height = image.width, image.height
    original_image = image.copy()
    width, height = image.width, image.height
    image = image.resize((width, height), Image.ANTIALIAS)
    canvas.config(width=width, height=height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

def update_image():

    global file_path, original_width, original_height, original_image
   
    image = Image.open(file_path)
    original_width, original_height = image.width, image.height
    original_image = image.copy()
    width, height = image.width, image.height
    image = image.resize((width, height), Image.ANTIALIAS)
    canvas.config(width=width, height=height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")


def change_color():
    global pen_color
    pen_color = colorchooser.askcolor(title="Select Pen Color")[1]


def change_size(size):
    global pen_size
    pen_size = size


def draw(event):
    global ovals
    global original_image

    x1, y1 = (event.x - pen_size), (event.y - pen_size)
    x2, y2 = (event.x + pen_size), (event.y + pen_size)
    canvas.create_oval(x1, y1, x2, y2, fill=pen_color, outline='')

    oval_coords = (x1, y1, x2, y2)
    ovals.append(oval_coords)


def save_drawing():
    global ovals, original_image, pen_color

    draw = ImageDraw.Draw(original_image)
     # Draw the ovals on the image
    for oval in ovals:
        x1, y1, x2, y2 = oval
        draw.ellipse((x1, y1, x2, y2), fill=pen_color)
        print(oval)

    # Save the modified image
    original_image.save(file_path)
  

    

def clear_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, image=canvas.image, anchor="nw")


def apply_filter(filter):
    image = Image.open(file_path)
    width, height = image.width, image.height
    image = image.resize((width, height), Image.ANTIALIAS)
    if filter == "Black and White":
        image = ImageOps.grayscale(image)
    elif filter == "Blur":
        image = image.filter(ImageFilter.BLUR)
    elif filter == "Sharpen":
        image = image.filter(ImageFilter.SHARPEN)
    elif filter == "Smooth":
        image = image.filter(ImageFilter.SMOOTH)
    elif filter == "Emboss":
        image = image.filter(ImageFilter.EMBOSS)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")


def zoom_in():
    global zoom_scale
    zoom_scale += 0.1
    apply_zoom()


def zoom_out():
    global zoom_scale
    if zoom_scale > 0.1:
        zoom_scale -= 0.1
        apply_zoom()


def apply_zoom():
    image = Image.open(file_path)
    width = int(original_width * zoom_scale)
    height = int(original_height * zoom_scale)
    image = image.resize((width, height), Image.ANTIALIAS)

    canvas.config(width=width, height=height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")


def pan_left():
    global pan_x
    pan_x -= 10
    apply_pan()


def pan_right():
    global pan_x
    pan_x += 10
    apply_pan()


def pan_up():
    global pan_y
    pan_y -= 10
    apply_pan()


def pan_down():
    global pan_y
    pan_y += 10
    apply_pan()


def apply_pan():
    canvas.delete("all")
    x = -pan_x
    y = -pan_y
    image = Image.open(file_path)
    width = int(original_width * zoom_scale)
    height = int(original_height * zoom_scale)
    image = image.resize((width, height), Image.ANTIALIAS)

    canvas.config(width=width, height=height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(x, y, image=image, anchor="nw")

def start_drawing(event):
    if mode == "draw":
        canvas.bind("<B1-Motion>", draw)

def stop_drawing(event):
    if mode == "draw":
        canvas.unbind("<B1-Motion>")

def crop_start(event):
    if mode == "crop":
        global crop_start_x, crop_start_y, crop_rectangle
        crop_start_x, crop_start_y = event.x, event.y
        crop_rectangle = canvas.create_rectangle(crop_start_x, crop_start_y, crop_start_x, crop_start_y, outline='red')

def crop_end(event):
    if mode == "crop":
        global crop_rectangle, original_image
        crop_end_x, crop_end_y = event.x, event.y
        x1 = min(crop_start_x, crop_end_x)
        y1 = min(crop_start_y, crop_end_y)
        x2 = max(crop_start_x, crop_end_x)
        y2 = max(crop_start_y, crop_end_y)

        x1 = int(x1 / zoom_scale) - pan_x
        y1 = int(y1 / zoom_scale) - pan_y
        x2 = int(x2 / zoom_scale) - pan_x
        y2 = int(y2 / zoom_scale) - pan_y

        canvas.delete(crop_rectangle)
        crop_rectangle = None  # Reset the crop rectangle

        # Crop the original image
        cropped_image = original_image.crop((x1, y1, x2, y2))

        cropped_image2save = original_image.crop((x1, y1, x2, y2))

        # Save the cropped image to the original image file
        cropped_image2save.save(file_path)

        # Update the canvas image with the cropped version
        width, height = original_image.width, original_image.height
        cropped_image = cropped_image.resize((width, height), Image.ANTIALIAS)
        cropped_image = ImageTk.PhotoImage(cropped_image)
        canvas.image = cropped_image
        canvas.create_image(0, 0, image=cropped_image, anchor="nw")


def analize():
    im_width = 256
    im_height = 256




    model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': -0.8741, 'iou': 0.7798, 'dice_coef': 0.8741})


    img = cv2.imread(file_path)
    img = cv2.resize(img ,(im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred) > .5)
    plt.title('Prediction')
    plt.show()



def set_mode(new_mode):
    global mode
    mode = new_mode

    if mode == "draw":
        canvas.config(cursor="")
        canvas.unbind("<Button-1>")
        canvas.unbind("<ButtonRelease-1>")
        canvas.bind("<Button-1>", start_drawing)
        canvas.bind("<ButtonRelease-1>", stop_drawing)
    elif mode == "crop":
        canvas.config(cursor="cross")
        canvas.unbind("<Button-1>")
        canvas.unbind("<ButtonRelease-1>")
        canvas.bind("<Button-1>", crop_start)
        canvas.bind("<ButtonRelease-1>", crop_end)


def revert_image():
    global original_image
    canvas.delete("all")
    width, height = original_image.width, original_image.height
    original_image = original_image.resize((width, height), Image.ANTIALIAS)
    original_image.save(file_path)
    original_image = ImageTk.PhotoImage(original_image)
    canvas.image = original_image
    canvas.create_image(0, 0, image=original_image, anchor="nw")

    update_image()



    



left_frame = tk.Frame(root, width=200, height=600, bg="white")
left_frame.pack(side="left", fill="y")

canvas = tk.Canvas(root, width=750, height=600)
canvas.pack()



image_button = tk.Button(left_frame, text="Add Image", command=add_image, bg="white")
image_button.pack(pady=15)

draw_button = tk.Button(left_frame, text="Draw", command=lambda: set_mode("draw"), bg="white")
draw_button.pack(pady=10)
draw_button.bind("<Button-1>", start_drawing)
draw_button.bind("<ButtonRelease-1>", stop_drawing)

color_button = tk.Button(left_frame, text="Change Pen Color", command=change_color, bg="white")
color_button.pack(pady=5)

pen_size_frame = tk.Frame(left_frame, bg="white")
pen_size_frame.pack(pady=5)

pen_size_1 = tk.Radiobutton(pen_size_frame, text="Small", value=3, command=lambda: change_size(3), bg="white")
pen_size_1.pack(side="left")

pen_size_2 = tk.Radiobutton(pen_size_frame, text="Medium", value=5, command=lambda: change_size(5), bg="white")
pen_size_2.pack(side="left")
pen_size_2.select()

pen_size_3 = tk.Radiobutton(pen_size_frame, text="Large", value=7, command=lambda: change_size(7), bg="white")
pen_size_3.pack(side="left")

clear_button = tk.Button(left_frame, text="Clear", command=clear_canvas, bg="#FF9797")
clear_button.pack(pady=10)

filter_label = tk.Label(left_frame, text="Select Filter", bg="white")
filter_label.pack()
filter_combobox = ttk.Combobox(left_frame, values=["Black and White", "Blur", "Emboss", "Sharpen", "Smooth"])
filter_combobox.pack()

filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))

save_drawing_button = tk.Button(left_frame, text="Save_Drawing", command=save_drawing, bg="white")
save_drawing_button.pack(pady=5)

zoom_in_button = tk.Button(left_frame, text="Zoom In", command=zoom_in, bg="white")
zoom_in_button.pack(pady=5)

zoom_out_button = tk.Button(left_frame, text="Zoom Out", command=zoom_out, bg="white")
zoom_out_button.pack(pady=5)

pan_left_button = tk.Button(left_frame, text="←", command=pan_left, bg="white")
pan_left_button.pack(pady=5)

pan_right_button = tk.Button(left_frame, text="→", command=pan_right, bg="white")
pan_right_button.pack(pady=5)

pan_up_button = tk.Button(left_frame, text="↑", command=pan_up, bg="white")
pan_up_button.pack(pady=5)

pan_down_button = tk.Button(left_frame, text="↓", command=pan_down, bg="white")
pan_down_button.pack(pady=5)

crop_button = tk.Button(left_frame, text="Crop", command=lambda: set_mode("crop"), bg="white")
crop_button.pack(pady=10)
crop_button.bind("<Button-1>", crop_start)
crop_button.bind("<ButtonRelease-1>", crop_end)


save_button = tk.Button(left_frame, text="Save Image", command=lambda: save_image(), bg="white")
save_button.pack(pady=10)

revert_button = tk.Button(left_frame, text="Revert", command=revert_image, bg="white")
revert_button.pack(pady=10)

analize = tk.Button(left_frame, text="ANALYZE THE TUMOR", command=analize, bg="white")
analize.pack(pady=10)


def toggle_mode(new_mode):
    global mode
    mode = new_mode
    if mode == "draw":
        canvas.bind("<Button-1>", draw)
        canvas.bind("<B1-Motion>", draw)
        canvas.config(cursor="pencil")
    elif mode == "crop":
        canvas.bind("<Button-1>", crop_start)
        canvas.bind("<ButtonRelease-1>", crop_end)
        canvas.config(cursor="crosshair")


def save_image():
    image = Image.open(file_path)
    image.save("output.png")
    print("Image saved as 'output.png'")


root.mainloop()
