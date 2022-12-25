from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import scipy.ndimage 
import scipy.ndimage as snd

def maxmin_norm(I):
    min_image = min(I.flatten())
    Inorm = I-min_image
    max_image = max(Inorm.flatten())
    Inorm = Inorm/max_image
    Inorm = Inorm*255
    return Inorm

def main():
    
    im = np.array(Image.open('./images/png/tiger.png').convert('L'))
    
    im_filt = scipy.ndimage.filters.gaussian_filter(im,2)

    # Türev islemi ile eksili degerler olusabileceginden double cevrimi gerekli
    im_filt = np.float64(im_filt)

    yf = np.array([[-1],[1]])
    xf = yf.transpose()
    print(xf)
    print(yf)

    im_cor_x = snd.correlate(im_filt, xf, mode='constant')
    im_cor_y = snd.correlate(im_filt, yf, mode='nearest')

    sy = scipy.ndimage.sobel(im_filt, axis=0, mode='constant') 
    sx = scipy.ndimage.sobel(im_filt, axis=1, mode='constant') 
    sob = np.hypot(sx, sy)
    print(sob.max())
    sob = sob>(np.max(sob.flatten())*0.1)
    sob = np.uint8(sob)*255

    im_con = np.concatenate((maxmin_norm(sx), maxmin_norm(sy)), axis=1)
    im_con2 = np.concatenate((maxmin_norm(im_cor_x), maxmin_norm(im_cor_y)), axis=1)

    # plt.show()
    # im_con = np.concatenate((sx, sy), axis=1)
    # imgplot = plt.imshow(im_con, cmap='gray') 
    # plt.show()
    # print(np.max(sob))
    # print(np.min(sob))
    
    imgplot = plt.imshow(im_con2, cmap='gray') 
    plt.show()
    imgplot = plt.imshow(im_con, cmap='gray') 
    plt.show()
    imgplot = plt.imshow(sob, cmap='gray') 
    plt.show()
    

if __name__ == '__main__':
    main()