import numpy as np
import matplotlib.pyplot as plt
import cv2

# With jupyter notebook uncomment below line
# %matplotlib inline
# This plots figures inside the notebook


def plot_multiple(img1, img2):
    """
    Converts an image from BGR to RGB and plots
    """

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Image 1")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Image 2")
    ax[1].axis("off")

    plt.show()


def compute_orb_keypoints(img):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors.
    """

    # create orb object
    orb = cv2.ORB_create()

    # set parameters
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
    # number of randomly selected points for comparison to compute BRIEF histogram bins
    # can be 2,3,4
    orb.setWTA_K(2)

    # detect keypoints
    kp = orb.detect(img, None)

    # for detected keypoints compute descriptors.
    kp, des = orb.compute(img, kp)
    return img, kp, des


def draw_keyp(img, kp):
    """
    Takes image and keypoints and plots on the same images
    Does not display it.
    """
    cv2.drawKeypoints(img, kp, img, color=(255, 0, 0), flags=2)
    return img


def plot_img(img, figsize=(12, 8)):
    """
    Plots image using matplotlib for the given figsize
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # image need to be converted to RGB format for plotting
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ORB keypoints")
    plt.show()


def main():
    # read images
    # filename1 = './images/jpg/office_3.jpg'
    # filename2 = './images/jpg/office_5.jpg'
    filename1 = "./images/jpg/hands1.jpg"
    filename2 = "./images/jpg/hands2.jpg"
    # load image
    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)
    # compute ORB keypoints
    img1, kp1, des1 = compute_orb_keypoints(image1)
    img2, kp2, des2 = compute_orb_keypoints(image2)
    # draw keypoints on images
    img1 = draw_keyp(img1, kp1)
    img2 = draw_keyp(img2, kp2)
    # plot images with keypoints
    plot_multiple(img1, img2)


if __name__ == "__main__":
    main()
