from __future__ import print_function

from keras.datasets import mnist
import matplotlib.pyplot as plt
from sift_features import compute_sift_features
import cv2
import numpy as np

# Download and load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# to know the size of data
print("Train data shape:", x_train.shape, "Test data shape:", x_test.shape)
# plot sample image
idx = 0
print("Label:", y_train[idx])
plt.imshow(x_train[idx], cmap="gray")
plt.axis("off")
plt.show()
img = cv2.cvtColor(np.asarray(x_train[idx]), cv2.COLOR_RGB2BGR)
compute_sift_features(img)

# image = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))
