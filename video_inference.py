import os
import cv2

# Creates a segmented video output from the already segmented images

# Source directory
dir_path = './inference_output'

# List of image paths
images = []

for f in os.listdir(dir_path):

    if f.endswith(ext):
        images.append(f)

image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)

# unpack the h,w,c variables from frame.shape
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 10.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame)

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

# After the loop is finished - releases the video
out.release()
