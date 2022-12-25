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

im = np.float32(im)

s_log = snd.filters.gaussian_laplace(im, (5,5))
s_laplace = snd.filters.laplace(im, mode='constant')

T = 0.8*np.sum(np.abs(s_log.flatten()))/s_log.size
s_log_thresh = np.abs(s_log)>T

T = 0.8*np.sum(np.abs(s_laplace.flatten()))/s_laplace.size
s_laplace_thresh = np.abs(s_laplace)>T

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,\
     ncols=2, figsize=(8, 8),sharex=True, sharey=True)
ax1.imshow(maxmin_norm(s_log), cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Laplacian of Gaussian', fontsize=20)
ax2.imshow(maxmin_norm(np.uint(s_log_thresh)), cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Laplacian of Gaussian Thresholded', fontsize=20)
ax3.imshow(maxmin_norm(s_laplace), cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Laplacian', fontsize=20)
ax4.imshow(maxmin_norm(np.uint(s_laplace_thresh)), cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title('Laplacian Thresholded', fontsize=20)
fig.tight_layout()
plt.show()
