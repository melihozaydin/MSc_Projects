import cv2 

# loads and read an image from path to file
img =  cv2.imread('./images/flowers1.png')

# convert the color to lab 
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

# displays previous image 
cv2.imshow("Image",lab)

# keeps the window open untill a key is pressed
cv2.waitKey(0)

# clears all window buffers
cv2.destroyAllWindows()
