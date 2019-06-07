import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

import numpy as np
import matplotlib.pyplot as plt
import itertools

ia.seed(1)

#parameters
palette = {(1,64,128):0,
           (3,143,255):1,
           (2,255,128):2,
           (0,0,0):3}

img_path = './dataset/training/images/'
label_path = './dataset/training/labels/'

image_shape = (512, 288)
width = image_shape[0]
height = image_shape[1]
num_classes = 4
batch_size=1
do_augment = False

# outer functions
from data_loader import get_pairs, input_image_array, input_label_array, generator

seq = [None]

def load_aug():

    seq[0] = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=True)

def _augment_seg(img, seg):
    if seq[0] is None:
        load_aug()

# deterministic augmentation
    aug_det = seq[0].to_deterministic()

    image_aug = aug_det.augment_image( img )

    seg_aug = SegmentationMapOnImage(seg , nb_classes=np.max(seg)+1 , shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps( seg_aug )
    segmap_aug = segmap_aug.get_arr_int(background_class_id=3)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.imshow(image_aug)
    ax2.imshow(segmap_aug)
    plt.show()


# image + label can be augmented in one line also
# image_aug, segmap_aug = aug_det(image=img, segmentation_maps=segmap)

def try_n_times(fn ,n ,*args ,**kargs):

	attempts = 0

	while attempts < n:
		try:
			return fn(*args , **kargs)
		except Exception as e:
			attempts += 1

	return fn(*args , **kargs)

def augment_seg(img_path, label_path): # kicserélni a kezdeti argumentumokra, return passzoljon
    input_pairs = get_pairs(img_path, label_path)
    iterate_pairs = itertools.cycle(input_pairs)
    im, lab = next(iterate_pairs)
    im_readin = next(iter(im))
    seg_readin = next(iter(lab))

    img = imageio.imread(im_readin, as_gray=False, pilmode="RGB")
    seg = input_label_array(seg_readin, width, height, num_classes, palette, do_augment=do_augment)
    return try_n_times(_augment_seg, 10, img, seg)

augment_seg(img_path, label_path)
