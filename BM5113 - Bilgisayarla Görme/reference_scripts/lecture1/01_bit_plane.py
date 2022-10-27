from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

im = np.array(Image.open('./images/png/kobi.png').convert('L'))

im1 = im & 192
im2 = im & 128

im_trans = np.concatenate((im, im1, im2), axis=1)

imgplot = plt.imshow(im1, cmap='gray')
plt.show()
imgplot = plt.imshow(im2, cmap='gray')
plt.show()
# original image
imgplot = plt.imshow(im_trans, cmap='gray')  
plt.show()