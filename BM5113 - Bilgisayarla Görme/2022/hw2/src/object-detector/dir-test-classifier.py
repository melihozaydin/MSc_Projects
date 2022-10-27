# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
import imutils
import glob
import ntpath
import os
import numpy as np
from nms import nms
from config import *

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def pyramid(image, scale, minSize):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        #compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        #  size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--path", help="Path to the test directory", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=float)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
    args = vars(parser.parse_args())

    path = args["path"]
    downscale = args['downscale']
    visualize_det = args['visualize']

    # Load the classifier
    clf = joblib.load(model_path)

    for im_path in glob.glob(os.path.join(path, "*")):
        # Read the image
        im = imread(im_path, as_grey=True)
        # List to store the detections
        detections = []
        # The current scale of the image
        scale = 0
        # Downscale the image and iterate
        for im_scaled in pyramid(image=im, scale=downscale, minSize=min_wdw_sz):
            # for im_scaled in pyramid_gaussian(image=im, downscale=downscale):
            # This list contains detections at the current scale
            cd = []
            # If the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations.
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                fd = hog(im_window, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
                fd = np.array(fd).reshape(1,(len(fd)))
                pred = clf.predict(fd)
                confidence_score = clf.decision_function(fd)
                if pred == 1 and confidence_score >= detection_threshold:
                    print "Detection:: Location -> ({}, {})".format(x, y)
                    print "Scale ->  {} | Confidence Score {} \n".format(scale, confidence_score)
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)), confidence_score,
                         int(min_wdw_sz[0] * (downscale ** scale)),
                         int(min_wdw_sz[1] * (downscale ** scale))))
                    # cd.append(detections[-1])
                    cd.append((x, y, confidence_score,
                               int(min_wdw_sz[0] * (downscale ** scale)),
                               int(min_wdw_sz[1] * (downscale ** scale))))
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _ in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                  im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress", clone)
                    cv2.waitKey(30)
            # Move the the next scale
            scale += 1
        if visualize_det:
            # Display the results before performing NMS
            clone_before_nms = im.copy()
            clone = im.copy()
            for (x_tl, y_tl, _, w, h) in detections:
                # Draw the detections
                cv2.rectangle(clone_before_nms, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
            cv2.imshow("Raw Detections before NMS", clone_before_nms)
            cv2.waitKey()

        # Perform Non Maxima Suppression
        detections = nms(detections, nms_threshold)

        # Display the results after performing NMS
        image = cv2.cvtColor(imread(im_path), cv2.COLOR_BGR2RGB)
        for (x_tl, y_tl, _, w, h) in detections:
            # Draw the detections
            cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
        cv2.imwrite("C:/Users/Equipo2/Desktop/detectadas/{}".format(ntpath.basename(im_path)), image)
        if visualize_det:
            cv2.imshow("Final Detections after applying NMS", image)
            cv2.waitKey()








