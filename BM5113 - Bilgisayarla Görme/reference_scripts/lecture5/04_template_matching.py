from PIL import Image
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt

im = np.array(Image.open('./images/jpg/car1.jpg'))
print(im.shape)
im_trim1 = im[1000:1500, 2600:2900]
print(im_trim1.shape)
imgplot = plt.imshow(im_trim1)  
plt.show()
pil_img = Image.fromarray(im_trim1)
pil_img.save('./images/jpg/car1_trimmed.jpg')

img = cv2.imread('./images/jpg/car1.jpg',0)
img2 = img.copy()
template = cv2.imread('./images/jpg/car1_trimmed.jpg',0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    print(method)
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 5)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()