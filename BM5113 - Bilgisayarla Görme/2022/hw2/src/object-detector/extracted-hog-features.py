# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import argparse as ap
import matplotlib.pyplot as plt
import glob
import os
import cv2
from config import *
from skimage.feature import hog
from skimage import data, color, exposure

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-n', "--negpath", help="Path to negative images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]
	
    des_type = args["descriptor"]

    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)

    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        image = imread(im_path, as_grey=True)
        if des_type == "HOG":
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                                cells_per_block=(2, 2), visualise=True, normalise=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            ax1.axis('off')
            ax1.imshow(image, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            ax1.set_adjustable('box-forced')

            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            ax1.set_adjustable('box-forced')
            plt.show()
            break