# imported U-Net model from model.py
from model import Unet

import cv2
import numpy as np
import os
import time

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# training on multiple GPUs
from tensorflow.keras.utils import multi_gpu_model

# Parameters and source directories

image_shape = (512, 288)
width = image_shape[0]
height = image_shape[1]
num_classes = 4

img_dir = "./dataset/training/images/"
label_dir = "./dataset/training/labels/"

# Color palette

palette = {(128,64,1):0,
           (255,143,3):1,
           (128,255,2):2,
           (0,0,0):3}

# PREPROCESSING

def input_image_array(path, width, height):
    img = cv2.imread(path, 1)
    img_array = np.float32(cv2.resize(img, (width, height))) / 255.0
    return img_array

def input_label_array(path, width, height, num_classes, color_codes):
    label = cv2.imread(path)
    label = cv2.resize(label, (width, height))

    int_array = np.ndarray(shape=(height, width), dtype=int)
    int_array[:,:] = 0

    # rgb to integer
    for rgb, idx in color_codes.items():
        int_array[(label==rgb).all(2)] = idx

    one_hot_array = np.zeros((height, width, num_classes))

    # one-hot encoding
    for c in range(num_classes):
        one_hot_array[:, :, c] = (int_array == c).astype(int)

    return one_hot_array

# lists to append input images and labels
X = []
y = []

images = os.listdir(img_dir)
images.sort()
labels = os.listdir(label_dir)
labels.sort()

for img, label in zip(images, labels):
    X.append(input_image_array(img_dir + img, width, height))
    y.append(input_label_array(label_dir + label, width, height, num_classes, palette))

# input layer takes NumPy Arrays
X, y = np.array(X), np.array(y)

# TRAINING
filters = 64

unet = Unet(height, width, num_classes, filters)
#unet_parallel = multi_gpu_model(unet, 1)
#unet_parallel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

unet.summary()

# naming for TensorBoard
NAME = "unet-drive-scene-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
checkpoint = ModelCheckpoint(mode='max', filepath='checkpoints/best_outcome.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
early_stop = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
callback_list = [tensorboard, checkpoint, early_stop]

# try it with fit_generator
backprop = unet.fit(X, y, validation_split=0.1, batch_size=16, epochs=200, callbacks=callback_list)

unet.save_weights("weights-drive-scene.h5", overwrite=True)
unet.save("model-drive-scene.model", overwrite=True)
