from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import math


# # read image to array
# im = np.array(Image.open('./images/eiffel2-1.jpg').convert('L'))
# # generate a new figure
# figure()
# # don’t use colors
# gray()
# # show contours with origin upper left corner
# contour(im, origin='image')
# axis('equal')
# axis('off')

# figure()
# hist(im.flatten(),128)
# show()

## NUMPY IMHIST

def imresize(im,sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    imhist_eq,bins = histogram(im2.flatten(),nbr_bins,normed=True)
    return im2.reshape(im.shape), imhist, cdf, imhist_eq

def compute_average(imlist):
    """ Compute the average of a list of images. """
    # open first image and make into array of type float
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped') 
    averageim /= len(imlist)

    # return average as uint8
    return array(averageim, 'uint8')


# read image to array
im = np.array(Image.open('./images/jpg/trailer.jpg').convert('L'))
im_eq, imhist, cdf, imhist_eq = histeq(im)
im_list = np.concatenate((im, im_eq), axis=1)
plt.plot(imhist)
plt.show()
plt.plot(cdf)
plt.show()
plt.plot(imhist_eq)
plt.show()
imgplot = plt.imshow(im_list, cmap='gray')  
plt.show()