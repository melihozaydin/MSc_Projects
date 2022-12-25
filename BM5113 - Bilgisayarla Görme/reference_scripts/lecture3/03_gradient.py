from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage as snd

im = np.array(Image.open('./images/jpg/car1.jpg').convert('L'))
filt = np.array([[0,1,0],[1,1,1],[0,1,0]])
filt = 1/5*filt
print(type(filt))
im_cor = snd.correlate(im, filt, mode='constant')
im_conv = snd.convolve(im, filt, mode='nearest')
# Gaussian Filtering
im_gauss = snd.filters.gaussian_filter(im,3)
im_gauss2 = snd.gaussian_filter(im,3)
# im_list = np.concatenate((im, im_cor, im_conv, im_gauss,im_gauss2), axis=1)
# imgplot = plt.imshow(im_list, cmap='gray') 
# plt.show()
im = np.float64(im)

sy = snd.filters.sobel(im, axis=0, mode='constant') 
sx = snd.filters.sobel(im, axis=1, mode='constant') 

sob = np.hypot(sx, sy)
sob = sob.astype(np.uint8)

im_list1 = np.concatenate((sx, sy), axis=1)
imgplot = plt.imshow(im_list1, cmap='gray') 
plt.show()
im_list2 = np.concatenate((im, sob), axis=1)
imgplot = plt.imshow(im_list2, cmap='gray') 
plt.show()

