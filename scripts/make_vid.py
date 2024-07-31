import cv2
import os

# Define the path to the images and the output video
image_folder = 'demo/imgs/img1'
video_name = 'demo/dancer_demo.mp4'

# Get the list of images
images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

# Read the first image to get the dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))

# Write the images to the video
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release the video writer object
cv2.destroyAllWindows()
video.release()