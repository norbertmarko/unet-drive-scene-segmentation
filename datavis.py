import glob
import numpy as np
import cv2
import random
import itertools
import matplotlib.pyplot as plt

palette = {(1,64,128):0,
           (3,143,255):1,
           (2,255,128):2,
           (0,0,0):3}

from data_loader import get_pairs, input_image_array, input_label_array
# import augmenter

img_path = './dataset/training/images/'
label_path = './dataset/training/labels/'
image_shape = (512, 288)
width = image_shape[0]
height = image_shape[1]
num_classes = 4
do_augment = False

def visualize_segmentation_dataset( images_path , segs_path ,  n_classes , do_augment ):

    img_seg_pairs = get_pairs( images_path , segs_path )

    print("Press any key to navigate. ")
    for im_fn , seg_fn in img_seg_pairs :

        im_readin = next(iter(im_fn))
        seg_readin = next(iter(seg_fn))
        img = input_image_array(im_readin, width, height, do_augment)
        seg = input_label_array(seg_readin, width, height, n_classes, palette, do_augment=do_augment)

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.imshow(img)
        ax2.imshow(seg)
        #ax3.imshow()
        #ax4.imshow()
        plt.show()

tryout = visualize_segmentation_dataset(img_path, label_path, num_classes, do_augment)
