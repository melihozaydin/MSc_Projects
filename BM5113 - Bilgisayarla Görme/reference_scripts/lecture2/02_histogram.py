from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt


# read image to array
im = np.array(Image.open('./images/jpg/eiffel2-1.jpg').convert('L'))
# generate a new figure
figure()
# don’t use colors
gray()
# show contours with origin upper left corner
contour(im, origin='image')
axis('equal')
axis('off')

imgplot = plt.imshow(im, cmap='gray')  
plt.show()

figure()
hist(im.flatten(),128)
show()
