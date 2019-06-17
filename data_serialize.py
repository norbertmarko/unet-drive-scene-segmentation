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

# get paths to images and labels
imagePaths = list(paths.list_images(img_path))
labelPaths = list(paths.list_images(label_path))
# shuffle
c = list(zip(imagePaths, labelPaths))
random.shuffle(c)
(imagePaths, labelPaths) = zip(*c)

# calculate input dimensions
img_dims = (len(imagePaths), height*width*3)
label_dims = (len(labelPaths), height*width*1)

# initialize lists to store the MEAN VALUES of the dataset
(R, G, B) = ([], [], [])

# Create an instance of the HDF5 serializer class
writer = HDF5writer(img_dims, label_dims, hdf5_path)

# initialize ProgressBar
widgets = ["Converting data into HDF5:", progressbar.Percentage(), " ",
progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

for i in np.arange(0, len(imagePaths), batchSize):
    batchImages = imagePaths[i:i + batchSize]
    batchLabels = labelPaths[i:i + batchSize]

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

print("[INFO] Dataset serialized successfully!")

print("[INFO] serializing means...")

# saving mean values
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(dataset_mean, "w")
f.write(json.dumps(D))
f.close()

print("[INFO] RGB mean values are calculated and saved across the dataset.")
