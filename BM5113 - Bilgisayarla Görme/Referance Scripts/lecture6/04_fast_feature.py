import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook



def plot_imgs(img1, img2):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('FAST points on Image (th=20)')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('FAST points on Image (th=20)')
    ax[1].axis('off')

    # ax[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))          
    # ax[2].set_title('FAST Points(th=15)')
    # ax[2].axis('off')    

    # ax[3].imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))          
    # ax[3].set_title('FAST Points(th=50)')
    # ax[3].axis('off')
    
    # plt.savefig('./images/04_fast_features_thres.png')

    plt.show()

def compute_fast_det(filename, is_nms=True, thresh = 10):

    img = cv2.imread(filename)
    
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create() #FastFeatureDetector()

    # find and draw the keypoints
    if not is_nms:
        fast.setNonmaxSuppression(0)

    fast.setThreshold(thresh)

    kp = fast.detect(img,None)
    cv2.drawKeypoints(img, kp, img, color=(0,0,255))
    
    return img


def main():
    # read an image 
    #img = cv2.imread('./images/flower.png')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filename1 = './images/jpg/yellowlily.jpg'
    filename2 = './images/jpg/yellowlily2.jpg'
    filename3 = './images/jpg/baby.jpg'
    # compute harris corners and display 
    img1 = compute_fast_det(filename1, thresh = 20)
    img2 = compute_fast_det(filename2, thresh = 20)
    img3 = compute_fast_det(filename3, thresh = 20)
    #img4 = compute_fast_det(filename, thresh = 10)
    
    # Do plot
    plot_imgs(img1, img2)
    plot_imgs(img1, img3)

if __name__ == '__main__':
    main()