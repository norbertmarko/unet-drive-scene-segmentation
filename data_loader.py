import glob
import os
import cv2
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from augment import augmenter
import imageio

image_shape = (512, 288)
width = image_shape[0]
height = image_shape[1]
num_classes = 4

palette = {(128,64,1):0,
           (255,143,3):1,
           (128,255,2):2,
           (0,0,0):3}

img_path = './dataset/training/images/'
label_path = './dataset/training/labels/'

batch_size=15

def get_pairs(img_path, label_path):

    pair = []

    img = glob.glob(os.path.join(img_path,"*.png"))
    label = glob.glob(os.path.join(label_path,"*.png"))

    pair.append((img, label))
    return pair

def input_image_array(path, width, height, do_augment=False):
    #img = cv2.imread(path, 1)
    img = imageio.imread(path, as_gray=False)
    img_array = np.uint8(cv2.resize(img, (width, height))) / 255.0 #better normalize, was np.float32

    if do_augment:
        img_array = augmenter(img_array)
    return img_array

def input_label_array(path, width, height, num_classes, color_codes, one_hot=True,
do_augment=False):
    #label = cv2.imread(path)
    label = imageio.imread(path, as_gray=False, pilmode="RGB")
    label = cv2.resize(label, (width, height))

    if do_augment:
        label[:,:,0] = augmenter(label[:,:,0])

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

def generator(img_path, label_path, batch_size, height, width, num_classes):

    input_pairs = get_pairs(img_path, label_path) # rewrite if param name changes
    random.shuffle(input_pairs)

    iterate_pairs = itertools.cycle(input_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, lab = next(iterate_pairs)

            appended_im = next(iter(im))
            appended_lab = next(iter(lab))

            X.append(input_image_array(appended_im, width, height))
            Y.append(input_label_array(appended_lab, width, height, num_classes, palette))

        yield (np.array(X), np.array(Y))
