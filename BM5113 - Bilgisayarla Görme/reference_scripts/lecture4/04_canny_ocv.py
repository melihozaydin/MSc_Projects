import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

def plot_cv_img(input_image, output_image):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(output_image)          
    ax[1].set_title('Edge Image')
    ax[1].axis('off')    
    
    # plt.savefig('../figures/xxx.png')

    plt.show()
    

def main():
    # read an image 
    img = cv2.imread('./images/png/flowers1.png')
    print(img.shape)
    img = cv2.resize(img, (1200,800))
    # cv2.imwrite('./images/flowers1_resized.png', img)
    blur = cv2.blur(img,(7,7))
    edges = cv2.Canny(img,100,200)
    # Do plot
    plot_cv_img(img, edges)

if __name__ == '__main__':
    main()