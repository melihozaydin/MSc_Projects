from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment
# GENERATION OF SINGLE COLOR IMAGES AND CONCATENATION
# im = np.array(Image.open('./images/jpg/car1.jpg'))
# im_R = im.copy()
# im_R[:, :, (1, 2)] = 0
# im_G = im.copy()
# im_G[:, :, (0, 2)] = 0
# im_B = im.copy()
# im_B[:, :, (0, 1)] = 0

# im_RGB = np.concatenate((im, im_R, im_G, im_B), axis=1)
# # im_RGB = np.hstack((im_R, im_G, im_B))
# # im_RGB = np.c_['1', im_R, im_G, im_B]
# imgplot = plt.imshow(im_RGB)  
# plt.show()

# COLOR REDUCTION
# im = np.array(Image.open('./jpg/car1.jpg').resize((256, 256)))

# im_32 = im // 32 * 32
# im_128 = im // 128 * 128

# im_dec = np.concatenate((im, im_32, im_128), axis=1)
# imgplot = plt.imshow(im_dec)  
# plt.show()

# GRAY-LEVEL TRANSFORMS
# im = np.array(Image.open('./jpg/car1.jpg').convert('L'))
# im2 = 255 - im #invert image
# im3 = (100.0/255) * im + 100 #clamp to interval 100...200
# im4 = 255.0 * (im/255.0)**2 #squared
# im5 = 255*(im >= 128) #thresholded
# im6 = im
# mask = im < 87
# im6[mask]=0

# im_trans = np.concatenate((im, im2, im3, im4, im5, im6), axis=1)

# imgplot = plt.imshow(im_trans, cmap='gray')  
# plt.show()

#TRIMMING AN IMAGE
# im = np.array(Image.open('./jpg/car1.jpg'))
# print(im.shape)
# im_trim1 = im[1000:1500, 2234:2890]
# print(im_trim1.shape)
# imgplot = plt.imshow(im_trim1)  
# plt.show()

# def trim(array, x, y, width, height):
#     return array[y:y + height, x:x+width]

# im_trim2 = trim(im, 1756, 2111, 524, 951)
# print(im_trim2.shape)
# imgplot = plt.imshow(im_trim2)  
# plt.show()

# im_cpy = im.copy()
# im_cpy[500:800, 2500:2800,:] = im[1500:1800, 3000:3300,:]
# imgplot = plt.imshow(im_cpy)  

# #plt.show()

# # from pylab import *
# # # some points
# x = [1000,1000,2000,2000]
# y = [2000,500,2000,500]
# # plot the points with red star-markers
# plt.plot(x,y,'r*')
# # line plot connecting the first three points
# plt.plot(x[:3],y[:3])

# plt.show()

#conda install -c anaconda scipy

# from scipy.ndimage import filters

# im = np.array(Image.open('./jpg/car1.jpg').convert('L'))
# im2 = filters.gaussian_filter(im,3)
# im3 = filters.gaussian_filter(im,5)
# im4 = filters.gaussian_filter(im,7)
# im5 = filters.gaussian_filter(im,9)
# im_ss = np.concatenate((im, im5), axis=1)

# imgplot = plt.imshow(im_ss, cmap='gray')  
# plt.show()

# NOISE ADDING
# im = np.array(Image.open('./jpg/car1.jpg').convert('L'))
# #a uniform distribution over [0, 1)
# dim1 = im.shape[0]
# dim2 = im.shape[1]
# temp = np.random.rand(dim1,dim2)
# print(temp.min())
# print(temp.max())
# mask1 = temp>0.8
# mask2 = temp<0.2
# im_ns1 = im.copy()
# im_ns1[mask1] = 255
# im_ns1[mask2] = 0
# noise = np.random.rand(dim1,dim2)*200-100
# im_ns2 = im+noise
# mask = im_ns2 > 255
# im_ns2[mask] = 255
# mask = im_ns2 < 0
# im_ns2[mask] = 0
# im_nss = np.concatenate((im, im_ns1, im_ns2), axis=1)
# imgplot = plt.imshow(im_nss, cmap='gray')  
# plt.show()