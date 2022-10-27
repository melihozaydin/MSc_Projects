from __future__ import print_function

from keras.datasets import cifar10
import matplotlib.pyplot as plt
from sift_features import compute_sift_features
from harrisCorner import compute_harris_corners
from orb_detections import compute_orb_keypoints, draw_keyp, plot_img
import cv2
import numpy as np

# Download and load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
# to know the size of data
print("Train data shape:", x_train.shape, "Test data shape:", x_test.shape)

# plot sample image
idx = 1800
print("Label:", labels[y_train[idx][0]])
plt.imshow(x_train[idx])
plt.axis("off")
plt.show()
# Harris Lecture 5
img = np.copy(x_train[idx])
compute_harris_corners(img)
# SIFT openCV
img = cv2.cvtColor(np.asarray(x_train[idx]), cv2.COLOR_RGB2BGR)
compute_sift_features(img)
# ORB openCV
img = cv2.cvtColor(np.asarray(x_train[idx]), cv2.COLOR_RGB2BGR)
img, kp, des = compute_orb_keypoints(img)
img = draw_keyp(img, kp)
plot_img(img)

idx = 100
print("Label:", labels[y_test[idx][0]])
plt.imshow(x_test[idx])
plt.axis("off")
plt.show()
# Harris openCV
img_c = cv2.cvtColor(np.asarray(x_test[idx]), cv2.COLOR_RGB2BGR)
img = cv2.cvtColor(np.asarray(x_test[idx]), cv2.COLOR_RGB2GRAY)
block_size = 2  # covariance matrix size
kernel_size = 5  # neighborhood kernel
k = 0.05  # parameter for harris corner score
thresh = 0.02
corners = cv2.cornerHarris(img, block_size, kernel_size, k)
img_c[corners > thresh * corners.max()] = [0, 0, 255]
plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
# SIFT openCV
img = cv2.cvtColor(np.asarray(x_test[idx]), cv2.COLOR_RGB2BGR)
compute_sift_features(img)
# ORB openCV
img = cv2.cvtColor(np.asarray(x_test[idx]), cv2.COLOR_RGB2BGR)
img, kp, des = compute_orb_keypoints(img)
img = draw_keyp(img, kp)
plot_img(img)
