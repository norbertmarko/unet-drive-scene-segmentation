# imported U-Net model from model.py
from model import Unet
from data_loader import generator

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

img_path = "./dataset/training/images/"
label_path = "./dataset/training/labels/"

batch_size=10
steps_per_epoch = len(os.listdir(img_path)) // batch_size
epochs = 2

# Color palette

palette = {(128,64,1):0,
           (255,143,3):1,
           (128,255,2):2,
           (0,0,0):3}

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

train_gen = generator(img_path, label_path, batch_size, height, width, num_classes)

# try it with fit_generator
#backprop = unet.fit(X, y, validation_split=0.1, batch_size=16, epochs=200, callbacks=callback_list)
backprop = unet.fit_generator(train_gen, steps_per_epoch, epochs)

unet.save_weights("weights-drive-scene.h5", overwrite=True)
unet.save("model-drive-scene.model", overwrite=True)
