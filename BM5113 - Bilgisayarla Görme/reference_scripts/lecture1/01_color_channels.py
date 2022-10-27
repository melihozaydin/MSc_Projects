from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment
# GENERATION OF SINGLE COLOR IMAGES AND CONCATENATION
im = np.array(Image.open('./images/jpg/car1.jpg'))
im_R = im.copy()
im_R[:, :, (1, 2)] = 0
im_G = im.copy()
im_G[:, :, (0, 2)] = 0
im_B = im.copy()
im_B[:, :, (0, 1)] = 0

im_RGB = np.concatenate((im, im_R, im_G, im_B), axis=1)
# im_RGB = np.hstack((im_R, im_G, im_B))
# im_RGB = np.c_['1', im_R, im_G, im_B]
imgplot = plt.imshow(im_RGB)  
plt.show()


