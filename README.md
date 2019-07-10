## U-Net drive scene segmentation 

######  Project for Szenergy

The purpose of this project is to provide a neural network capable of semantic segmentation for the Szenergy Team. This network will act as a groundwork for further deep learning solutions (e.g. using it for transfer learning )implemented in the autonomous vehicle of the team. It is a U-Net implementation from scratch using tf.keras, which will be trained on our custom relabeled cityscapes dataset.

Training results(separate test images) after 150 epochs, custom re-labeled cityscapes dataset, with 5 classes:

![Example 1](https://github.com/norbertmarko/unet-drive-scene-segmentation/blob/master/results.png)

###### Documentation:

Steps:
1. Serializing the data into HDF5 format
2. Setting the training parameters
3. Setting training details
4. Training


######1. Serializing the data into HDF5 format

The data_to_hdf5.py contains the class, that I wrote to help in the training/validation set serialization. This is imported in the data_serialize.py. You have to run the latter script, to convert your dataset into HDF5 format.
Your training data has to be in a folder named ’dataset’ in your root directory, and it must contain a ’training’ folder with two sub-directories – ’images’, ’labels’.
If you have a manually separated validation set ready, then put it into ’./dataset/validation’ with the same sub-directories as ’training’. 
If you want to separate a part of the training set for validation, then do not make / delete this folder in your dataset dir.

Mean values are calculated across the entire dataset, these are stored in a json string along with the serialized data in the hdf5_container directory.

######2. Setting the training parameters	

parameters.py contains the majority of the parameters you can adjust during training, but this will be updated in the future in case of added functionality and bugs.
Number of classes will be determined by the length of the palette you provide in the parameters script. Replace it with your RGB color – integer pairs in the provided format.

######3. Setting training details

You have options to use the following functionalities by setting the parameters:

-	One-hot Encoding / Integer-based Encoding
-	Mean subtraction – set the ’preprocessors’ value to [mp] (’mp’ in a list), or None if you do not wish to use it.
-	Data augmentation with the imgaug library – True/False – In case you set augmentation, you have to modify the augmentation.py inside the square brackets in seq[0] = iaa.Sequential([] part – instructions to use augmentation can be found ont he imgaug github page
-	Multi - GPU training – ’gpu_count’ is the number of your GPUs, if you give it a value of more than 1, it will automatically use a multi-GPU method for training
-	To set callbacks like TensorBoard, Checkpointing or EarlyStopping, modify the training.py fit_generator function by providing a list for the ’callbacks’ parameter (available callbacks: ’tensorboard’, ’checkpoint’, ’early_stop’)  - Tensorboard logs will be saved in the ’logs’ directory



######4. Training 

Training can be launched by running the training.py script. Models are saved in the saved_models directory and weights are saved in the saved_weights directory (if they do not exist, create them in root dir).

Things to add:

-	Image inference (inference.py) updated and automated
-	Video inference (video_inference.py) updated and automated
-	Data Visualization tool for seeing the results of preprocessing before starting training (there is a basic script uploaded but right now it is unusable)





