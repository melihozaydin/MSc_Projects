from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage as snd

def maxmin_norm(I):
    min_image = min(I.flatten())
    Inorm = I-min_image
    max_image = max(Inorm.flatten())
    Inorm = Inorm/max_image
    Inorm = Inorm*255
    return Inorm

im = np.array(Image.open('./images/jpg/trailer.jpg').convert('L'))
# Gaussian Filtering
im_gauss = snd.filters.gaussian_filter(im,3)

im = np.float32(im)
im_gauss = np.float32(im_gauss)

sy_sobel = snd.filters.sobel(im_gauss, axis=0, mode='constant') 
sx_sobel = snd.filters.sobel(im_gauss, axis=1, mode='constant') 
grad_mag = np.hypot(sx_sobel, sy_sobel)
grad_mag = maxmin_norm(grad_mag)

im_list2 = np.concatenate((im, grad_mag), axis=1)
imgplot = plt.imshow(im_list2, cmap='gray') 
plt.show()


sx_prewitt = snd.filters.prewitt(im_gauss, axis=1, mode='constant') 
sy_prewitt = snd.filters.prewitt(im_gauss, axis=0, mode='constant')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,\
     ncols=2, figsize=(8, 8),sharex=True, sharey=True)
ax1.imshow(maxmin_norm(sx_prewitt), cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Prewitt_x', fontsize=20)
ax2.imshow(maxmin_norm(sy_prewitt), cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Prewitt_y', fontsize=20)
ax3.imshow(maxmin_norm(sx_sobel), cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Sobel_x', fontsize=20)
ax4.imshow(maxmin_norm(np.uint(sy_sobel)), cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title('Sobel_y', fontsize=20)
fig.tight_layout()
plt.show()