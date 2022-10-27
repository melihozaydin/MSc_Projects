from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment


#conda install -c anaconda scipy

from scipy.ndimage import filters

im = np.array(Image.open('./images/jpg/car1.jpg').convert('L'))
im2 = filters.gaussian_filter(im,3)
im3 = filters.gaussian_filter(im,5)
im4 = filters.gaussian_filter(im,7)
im5 = filters.gaussian_filter(im,15)
im_ss = np.concatenate((im, im3, im5), axis=1)

imgplot = plt.imshow(im_ss, cmap='gray')  
plt.show()