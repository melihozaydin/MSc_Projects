from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage as snd
from scipy.ndimage import measurements,morphology

def maxmin_norm(I):
    min_image = min(I.flatten())
    Inorm = I-min_image
    max_image = max(Inorm.flatten())
    Inorm = Inorm/max_image
    Inorm = Inorm*255
    return Inorm

im = np.array(Image.open('./images/jpg/walls-l.jpg').convert('L'))
im = np.float32(im)
im = snd.filters.gaussian_filter(im,3)
sy = snd.filters.sobel(im, axis=0, mode='constant') 
sx = snd.filters.sobel(im, axis=1, mode='constant')
B1 = np.power(sy,2)
A1 = np.power(sx,2)
C1 = np.multiply(sx,sy)
A= snd.filters.gaussian_filter(A1,3)
B= snd.filters.gaussian_filter(B1,3)
C= snd.filters.gaussian_filter(C1,3)
#C = C.astype(np.uint)
k= 0.05
corner_det = np.multiply(A,B)-np.power(C,2)
corner_trc = A+B
cornerness = corner_det-k*np.power(corner_trc,2)
c_max = cornerness.max()
c_thresh = 0.01*c_max
I_cornerness = cornerness>c_thresh
# morphology - opening to separate objects better
se = np.array([[ 0, 1, 0],\
               [ 1, 1, 1],\
               [ 0, 1, 0]])
# binarization               
se = se>0 
I_cornerness_dilated = morphology.binary_dilation\
    (I_cornerness,se,iterations=11)

im_corner = np.zeros((im.shape[0], im.shape[1],3), dtype='uint8')
im_corner[:,:,0]=im.copy()
im_corner[:,:,1]=im.copy()
im_corner[:,:,2]=im.copy()

im_corner[I_cornerness_dilated]=[255,0,0]

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 8),
                                    sharex=True, sharey=True)

ax1.imshow(I_cornerness, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('I_cornerness', fontsize=20)

ax2.imshow(I_cornerness_dilated, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('I_cornerness_open', fontsize=20)

ax3.imshow(im_corner, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Image', fontsize=20)

fig.tight_layout()

plt.show(block=False)

# display autocorrelation matrix elements
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 8),
                                    sharex=True, sharey=True)

ax1.imshow(maxmin_norm(A), cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('A', fontsize=20)

ax2.imshow(maxmin_norm(B), cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('B', fontsize=20)

ax3.imshow(maxmin_norm(C), cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('C', fontsize=20)

fig.tight_layout()

plt.show()