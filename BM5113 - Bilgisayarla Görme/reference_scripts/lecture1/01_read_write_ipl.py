from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = np.array(Image.open('./images/jpg/car1.jpg'))
print(im.dtype)
# uint8
print(im.ndim)
# 3
print(im.shape)

imgplot = plt.imshow(im, cmap='gray')  
plt.show()

im_f = np.array(Image.open('./images/jpg/car2.jpg'), np.float64)
print(im_f.dtype)
# float64
print(im[256, 256])
# print RGB color
print(im[:, :, 0].min())

R = im[:, :, 0]
G = im[:, :, 1]
B = im[:, :, 2]

im_RGB = np.concatenate((R, G, B), axis=1)
imgplot = plt.imshow(im_RGB, cmap='gray')  
plt.show()

# print minimum R channel value

# pil_img = Image.fromarray(im)
# pil_img.save('./images/jpg/car1_copy.jpg')
# #save an image
# Image.fromarray(im).save('./images/jpg/car1_copy.jpg')
# #alternative usage
# print(im_f[:,:,:].min())
# print(im_f.max())
# pil_img_f = Image.fromarray(im_f.astype(np.uint8))
# pil_img_f.save('./images/jpg/car2_copy.jpg')
# #type conversion from float to uint