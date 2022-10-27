import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage
from scipy.ndimage import measurements,morphology

# load image and threshold to make sure it is binary
im = np.array(Image.open('./images/png/coins.png').convert('L'))
im_bin = (im>100)
labels, nbr_objects = measurements.label(im_bin)
print ("Number of objects:", nbr_objects)
imgplot = plt.imshow(im, cmap='gray') 
plt.show()
im_list = np.concatenate((im_bin, labels), axis=1)
imgplot = plt.imshow(im_list) 
plt.show()
# morphology - opening to separate objects better
se = np.array([[0, 0, 1, 0, 0],\
               [0, 1, 1, 1, 0],\
               [1, 1, 1, 1, 1],\
               [0, 1, 1, 1, 0],\
               [0, 0, 1, 0, 0]])

im_open = morphology.binary_closing(im_bin,se,iterations=2)
labels_open, nbr_objects_open = measurements.label(im_open)
im_list = np.concatenate((im_bin, im_open, labels_open), axis=1)
imgplot = plt.imshow(im_list) 
plt.show()
print ("Number of objects:", nbr_objects_open)
