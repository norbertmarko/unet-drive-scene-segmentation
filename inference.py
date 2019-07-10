import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# loading in the trained model
model = tf.keras.models.load_model("./saved_models/unet-drive-scene-1562336761.model")

# color map for segmentation

colors = np.array([[1, 64,128],
                   [3,143,255],
                   [2,255,128],
                   [255,140,0],
                   [0,  0,  0]])

# reshaping at the end to fit model dimensions

def prepare(path):
    img = cv2.imread(path)
    b,g,r = cv2.split(img)
    img_array_rgb = cv2.merge([r,g,b])
    img_array = cv2.resize(img_array_rgb, (512, 288))

    return img_array.reshape(-1, 288, 512, 3)

# path to the image you want to predict
image_path = './segnet_bayes_00001_input.png'
prepared_image = prepare(image_path)

prediction = model.predict([prepared_image])

# taking the depthwise argmax - choose given pixel from channel with the highest prob.
mask = np.argmax(prediction[0], axis=2)

# final segmentation result
colored_mask = colors[mask]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.imshow(prepared_image.reshape(288, 512, 3))
ax2.imshow(colored_mask)
plt.show()
