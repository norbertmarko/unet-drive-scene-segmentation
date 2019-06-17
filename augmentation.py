import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage

import numpy as np
import itertools

#parameters
from parameters import seed_value

ia.seed(seed_value)

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

    images = []
    labels = []

    image_aug = aug_det.augment_image( img )

    seg_aug = SegmentationMapOnImage(seg , nb_classes=np.max(seg)+1 , shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps( seg_aug )
    segmap_aug = segmap_aug.get_arr_int(background_class_id=3)

    return image_aug, segmap_aug

# image + label can be augmented in one line also
# image_aug, segmap_aug = aug_det(image=img, segmentation_maps=segmap)

def try_n_times(fn ,n ,*args ,**kwargs):

	attempts = 0

	while attempts < n:
		try:
			return fn(*args , **kwargs)
		except Exception as e:
			attempts += 1

	return fn(*args , **kwargs)
