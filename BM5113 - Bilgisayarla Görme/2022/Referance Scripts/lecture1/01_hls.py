import cv2 

# loads and read an image from path to file
img =  cv2.imread('./images/flowers1.png')

# convert the color to hls 
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# displays previous image 
cv2.imshow("Image",hls)

# keeps the window open untill a key is pressed
cv2.waitKey(0)

# clears all window buffers
cv2.destroyAllWindows()