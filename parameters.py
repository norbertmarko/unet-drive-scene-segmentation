import os
# Color Palette - RGB : Integer dictionary
# data_Serialize.py
palette = {(1,64,128):0,
           (3,143,255):1,
           (2,255,128):2,
           (255,140,0):3,
           (0,0,0):4}

# Paths
# data_serialize.py
img_path = './dataset/training/images/'
# data_serialize.py
label_path = './dataset/training/labels/'
# data_serialize.py
img_val_path = './dataset/validation/images/'
# data_serialize.py
label_val_path = './dataset/validation/labels/'
# data_serialize.py, training.py
hdf5_path ='./hdf5_container/serialized_data.hdf5'
# data_serialize.py, training.py
hdf5_val_path ='./hdf5_container/validation_data.hdf5'
# data_serialize.py
dataset_mean = './hdf5_container/mean_values.json'

# Image Parameters
# data_serialize.py
image_shape = (512, 288) # augmentation takes info directly from image
# data_serialize.py
width = image_shape[0]
# data_serialize.py
height = image_shape[1]
# number of unique classes depends on how many dictionary keys palette has
# data_loader.py
num_classes = len(palette) # seg_aug in augmentation.py uses: nb_classes=np.max(seg)+1

# Serialization
# data_serialize.py
bufferSize = None
batchSize=32

# Training Parameters
validation_split = 0.1
dataset_length = len(os.listdir(img_path))
# data_serialize.py
val_size = int( dataset_length * validation_split )
# augmentation.py
seed_value = 1
# training.py
batch_size = 2
# training.py
val_batch_size = 2
# training.py
val_steps = int(val_size // val_batch_size)
steps_per_epoch = int( dataset_length // batch_size )
# training.py
epochs = 150
# training.py
one_hot=True
# training.py
preprocessors=None # give a list in the training.py to change it
# training.py
do_augment=True
# training.py
gpu_count = 1
# training.py
show_summary=True
