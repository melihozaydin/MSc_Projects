from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment

# # NOISE ADDING
im = np.array(Image.open('./images/jpg/car1.jpg').convert('L'))
#a uniform distribution over [0, 1)
dim1 = im.shape[0]
dim2 = im.shape[1]
temp = np.random.rand(dim1,dim2)
print(temp.min())
print(temp.max())
mask1 = temp>0.8
mask2 = temp<0.2
im_ns1 = im.copy()
im_ns1[mask1] = 255
im_ns1[mask2] = 0
noise = np.random.rand(dim1,dim2)*200-100
im_ns2 = im+noise
mask = im_ns2 > 255
im_ns2[mask] = 255
mask = im_ns2 < 0
im_ns2[mask] = 0
im_nss = np.concatenate((im, im_ns1, im_ns2), axis=1)
imgplot = plt.imshow(im_nss, cmap='gray')  
plt.show()