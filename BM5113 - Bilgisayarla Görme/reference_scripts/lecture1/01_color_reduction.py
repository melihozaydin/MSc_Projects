from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment

# COLOR REDUCTION
im = np.array(Image.open('./images/jpg/car1.jpg').resize((256, 256)))

im_32 = im // 32 * 32
im_128 = im // 128 * 128

im_dec = np.concatenate((im, im_32, im_128), axis=1)
imgplot = plt.imshow(im_dec)  
plt.show()
