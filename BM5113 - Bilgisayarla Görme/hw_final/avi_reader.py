import numpy as np
import cv2
import matplotlib.pyplot as plt

# read video
cap = cv2.VideoCapture('./lecture7/7.avi')
ret1, frame1 = cap.read()
bg = np.mean(np.float32(frame1),axis=2)
while(cap.isOpened()):
    ret2, frame2 = cap.read()
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    frame2 = np.mean(np.float32(frame2), axis=2)
    fg = np.abs(frame2-bg)>10
    bg = 0.95*bg+frame2*0.05
    fg = np.uint8(fg)*255

    plt.imshow(fg,cmap=plt.cm.gray)  
    plt.draw()
    plt.show(block=False)    
    cv2.imshow('frame', rgb) 
    # video processing

    frame1 = frame2
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
 
 
cap.release()
cv2.destroyAllWindows()
 