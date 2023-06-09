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

#Set Parameters
im_width = 256
im_height = 256


model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': -0.1898, 'iou': 0.10, 'dice_coef': 0.1898})


img = cv2.imread("test1.tif")
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