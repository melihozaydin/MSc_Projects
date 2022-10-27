from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment

#TRIMMING AN IMAGE
im = np.array(Image.open('./images/jpg/car1.jpg'))
print(im.shape)
im_trim1 = im[1000:1500, 2234:2890]
print(im_trim1.shape)
imgplot = plt.imshow(im_trim1)  
plt.show()

def trim(array, x, y, width, height):
    return array[y:y + height, x:x+width]

im_trim2 = trim(im, 1756, 2111, 524, 951)
print(im_trim2.shape)
imgplot = plt.imshow(im_trim2)  
plt.show()

im_cpy = im.copy()
im_cpy[500:800, 2500:2800,:] = im[1500:1800, 3000:3300,:]
imgplot = plt.imshow(im_cpy)  

# # from pylab import *
# # some points
x = [1000,1000,2000,2000]
y = [2000,500,2000,500]
# plot the points with red star-markers
plt.plot(x,y,'r*')
# line plot connecting the first three points
plt.plot(x[:3],y[:3])

plt.show()