import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# loading in the trained model
model = tf.keras.models.load_model("./saved_models/model-drive-scene.model")

# color map for segmentation

colors = np.array([[1, 64,128],
                   [3,143,255],
                   [2,255,128],
                   [0,  0,  0]])

# reshaping at the end to fit model dimensions

def prepare(path):
    img = cv2.imread(path)
    b,g,r = cv2.split(img)
    img_array_rgb = cv2.merge([r,g,b])
    img_array = cv2.resize(img_array_rgb, (512, 288))

    return img_array.reshape(-1, 288, 512, 3)

# path to the image you want to predict
image_path = 'h003.jpg'

prediction = model.predict([prepare(image_path)])

# taking the depthwise argmax - choose given pixel from channel with the highest prob.
mask = np.argmax(prediction[0], axis=2)

# final segmentation result
colored_mask = colors[mask]

plt.imshow(colored_mask)
plt.show()
