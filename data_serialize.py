# encoding='utf-8'

#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import os
from imutils import paths
import imageio
import cv2
import random
import progressbar
import json

# classes
from data_to_hdf5 import HDF5writer

# neccessary parameters
from parameters import palette
from parameters import img_path, label_path, hdf5_path, dataset_mean
from parameters import image_shape, width, height, batchSize, bufferSize
from parameters import dataset_length, hdf5_val_path
from parameters import val_size, img_val_path, label_val_path

# get paths to images and labels
imagePaths = list(paths.list_images(img_path))
labelPaths = list(paths.list_images(label_path))
# shuffle
c = list(zip(imagePaths, labelPaths))
random.shuffle(c)
(imagePaths, labelPaths) = zip(*c)

# create or delete validation directory depending on needs
val_exist = os.path.isdir('./dataset/validation')

if val_exist:
    training_split_img = imagePaths
    training_split_label = labelPaths

    val_imagePaths = list(paths.list_images(img_val_path))
    val_labelPaths = list(paths.list_images(label_val_path))

    # shuffle
    c = list(zip(val_imagePaths, val_labelPaths))
    random.shuffle(c)
    (val_split_img, val_split_label) = zip(*c)

else:
    training_split_img = imagePaths[val_size:]
    training_split_label = labelPaths[val_size:]

    val_split_img = imagePaths[:val_size]
    val_split_label = labelPaths[:val_size]

# calculate input dimensions
img_dims = (len(training_split_img), height*width*3)
label_dims = (len(training_split_label), height*width*1)

v_img_dims = (len(val_split_img), height*width*3)
v_label_dims = (len(val_split_label), height*width*1)

# initialize lists to store the MEAN VALUES of the dataset
(R, G, B) = ([], [], [])

# Create an instance of the HDF5 serializer class
writer = HDF5writer(img_dims, label_dims, hdf5_path)

# TRAINING DATA
# initialize ProgressBar
widgets = ["Converting training data into HDF5:", progressbar.Percentage(), " ",
progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(training_split_img), widgets=widgets).start()

for i in np.arange(0, len(training_split_img), batchSize):
    batchImages = training_split_img[i:i + batchSize]
    batchLabels = training_split_label[i:i + batchSize]

    # lists to store the batches during loop
    batch_img = []
    batch_label = []

    for (j, imagePath) in enumerate(batchImages):
        img = imageio.imread(imagePath, as_gray=False, pilmode="RGB")
        img_array = cv2.resize(img, (512, 288))

        # mean claculation for actual image
        (b, g, r) = cv2.mean(img_array)[:3]
        R.append(r)
        G.append(g)
        B.append(b)

        img_array.reshape(-1, height, width, 3)

        batch_img.append(img_array)

    for (k, labelPath) in enumerate(batchLabels):
        label = imageio.imread(labelPath, as_gray=False, pilmode="RGB")
        label = cv2.resize(label, (width, height))

        int_array = np.ndarray(shape=(height, width), dtype=int)
        int_array[:,:] = 0

        # rgb to integer
        for rgb, idx in palette.items():
            int_array[(label==rgb).all(2)] = idx

        int_array.reshape(-1, height, width, 1)

        batch_label.append(int_array)

    batch_img = np.array(batch_img).reshape(-1, height*width*3)
    batch_label = np.array(batch_label).reshape(-1, height*width*1)

    # using the add() function from HDF5 writer class to serialize
    writer.add(batch_img, batch_label)

    # updating the ProgressBar
    pbar.update(i)

pbar.finish()
writer.close()

print("[INFO] TRAINING Dataset serialized successfully!")

print("[INFO] serializing means...")

# saving mean values
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(dataset_mean, "w")
f.write(json.dumps(D))
f.close()

print("[INFO] RGB mean values are calculated and saved across the dataset.")

val_writer = HDF5writer(v_img_dims, v_label_dims, hdf5_val_path)

# VALIDATION DATA
# initialize ProgressBar
widgets = ["Converting validation data into HDF5:", progressbar.Percentage(), " ",
progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(val_split_img), widgets=widgets).start()

for i in np.arange(0, len(val_split_img), batchSize):
    v_batchImages = val_split_img[i:i + batchSize]
    v_batchLabels = val_split_label[i:i + batchSize]

    # lists to store the batches during loop
    v_batch_img = []
    v_batch_label = []

    for (j, imagePath) in enumerate(v_batchImages):
        img = imageio.imread(imagePath, as_gray=False, pilmode="RGB")
        img_array = cv2.resize(img, (512, 288))

        img_array.reshape(-1, height, width, 3)

        v_batch_img.append(img_array)

    for (k, labelPath) in enumerate(v_batchLabels):
        label = imageio.imread(labelPath, as_gray=False, pilmode="RGB")
        label = cv2.resize(label, (width, height))

        int_array = np.ndarray(shape=(height, width), dtype=int)
        int_array[:,:] = 0

        # rgb to integer
        for rgb, idx in palette.items():
            int_array[(label==rgb).all(2)] = idx

        int_array.reshape(-1, height, width, 1)

        v_batch_label.append(int_array)

    v_batch_img = np.array(v_batch_img).reshape(-1, height*width*3)
    v_batch_label = np.array(v_batch_label).reshape(-1, height*width*1)

    # using the add() function from HDF5 writer class to serialize
    val_writer.add(v_batch_img, v_batch_label)

    # updating the ProgressBar
    pbar.update(i)

pbar.finish()
val_writer.close()

print("[INFO] VALIDATION Dataset serialized successfully!")
