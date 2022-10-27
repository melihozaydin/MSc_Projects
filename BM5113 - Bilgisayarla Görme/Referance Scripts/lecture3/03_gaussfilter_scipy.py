from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import scipy.ndimage 

def main():
    
    im = np.array(Image.open('./images/jpg/car1.jpg').convert('L'))
    
    im_filt = scipy.ndimage.filters.gaussian_filter(im,3)

    sy = scipy.ndimage.sobel(im_filt, axis=0, mode='constant') 
    sx = scipy.ndimage.sobel(im_filt, axis=1, mode='constant') 
    sob = np.hypot(sx, sy)


    im_con = np.concatenate((im, im_filt), axis=1)
    # imgplot = plt.imshow(im_con, cmap='gray')  

    # plt.show()
    # im_con = np.concatenate((sx, sy), axis=1)
    # imgplot = plt.imshow(im_con, cmap='gray') 
    # plt.show()
    # print(np.max(sob))
    # print(np.min(sob))
    sob = (sob/np.max(sob))*255
    imgplot = plt.imshow(np.uint8(sob>240)*255, cmap='gray') 
    plt.show()
    

if __name__ == '__main__':
    main()