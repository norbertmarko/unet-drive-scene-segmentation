# encoding='utf-8'

#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import json
import os
import time

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# imported U-Net model from model.py
from model import Unet

# classes
from generator_tools import MeanPreprocessor, HDF5_generator
# parameters
from parameters import dataset_mean, gpu_count, show_summary, val_steps
from parameters import hdf5_path, image_shape, num_classes, batch_size
from parameters import one_hot, do_augment, hdf5_val_path, val_batch_size
from parameters import height, width, num_classes, steps_per_epoch, epochs

# mean subtraction preparation
means = json.loads(open(dataset_mean).read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])

# training data generator
train_gen = HDF5_generator(hdf5_path, image_shape, num_classes, batch_size,
                           one_hot, preprocessors=[mp], do_augment=False)

# validation data generator
val_gen = HDF5_generator(hdf5_val_path, image_shape, num_classes,
                         val_batch_size, one_hot, preprocessors=[mp], do_augment=False)

# naming for TensorBoard
NAME = "unet-drive-scene-{}".format(int(time.time()))

# callbacks
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
checkpoint = ModelCheckpoint(mode='max', filepath='checkpoints/best_outcome.h5',
                   monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
early_stop = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
callback_list = [tensorboard, checkpoint, early_stop]

# define model


if gpu_count <= 1:
    print("[INFO] Training network with 1 GPU...")
    model = Unet(height, width, num_classes)
else:
    print("[INFO] Training with {} GPUs...".format(gpu_count))
    from tensorflow.keras.utils import multi_gpu_model

    with tf.device("/cpu:0"):
        model = Unet(height, width, num_classes, train=True)

    # making model parallel
    model = multi_gpu_model(model, gpu_count)
# initialize model and optimizer (needs to be added later)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if show_summary:
    model.summary()

    # backpropagation
    backprop = model.fit_generator(train_gen.generator(), steps_per_epoch, epochs,
                                            validation_data=val_gen.generator(), validation_steps=val_steps,
                                            callbacks=callback_list)

    # saving weights / model
    model.save_weights("{}.h5".format(NAME), overwrite=True)
    model.save("{}.model".format(NAME), overwrite=True)
