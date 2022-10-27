from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
# CTRL+K-C--> comment  CTRL+KU-->uncomment
# # GRAY-LEVEL TRANSFORMS
im = np.array(Image.open('./images/jpg/office_1.jpg').convert('L'))
im2 = 255 - im #invert image
im3 = (100.0/255) * im + 100 #clamp to interval 100...200

im4 = 255.0 * (im/255.0)**0.4 #squared

im5 = 255*(im >= 50) #thresholded
im6 = im
mask = im < 10
im6[mask]=0

im_trans = np.concatenate((im, im4), axis=1)

imgplot = plt.imshow(im_trans, cmap='gray')  
plt.show()