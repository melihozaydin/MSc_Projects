from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage as snd

im = np.array(Image.open('./images/jpg/trailer.jpg').convert('L'))
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

sy = snd.filters.sobel(im, axis=0, mode='constant') 
sx = snd.filters.sobel(im, axis=1, mode='constant') 

sob = np.hypot(sx, sy)
sob = sob.astype(np.uint8)


im_list2 = np.concatenate((im, sx, sy, sob), axis=1)
imgplot = plt.imshow(im_list2, cmap='gray') 
plt.show()


sx_prewitt = snd.filters.prewitt(im, axis=1, mode='constant') 
sy_prewitt = snd.filters.prewitt(im, axis=0, mode='constant')
s_log = snd.filters.gaussian_laplace(im, (5,5))
s_laplace = snd.filters.laplace(im, mode='constant')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),sharex=True, sharey=True)

ax1.imshow(sx_prewitt, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Prewitt_x', fontsize=20)

ax2.imshow(sy_prewitt, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Prewitt_y', fontsize=20)

ax3.imshow(s_log, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Laplacian of Gaussian', fontsize=20)

ax4.imshow(s_laplace, cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title('Laplacian', fontsize=20)

fig.tight_layout()

plt.show()
