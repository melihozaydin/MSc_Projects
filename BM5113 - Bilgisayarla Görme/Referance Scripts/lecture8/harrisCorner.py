from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage as snd
from scipy.ndimage import measurements, morphology
from skimage.color import rgb2gray


def compute_harris_corners(im_ori):
    im = rgb2gray(im_ori)
    im = im.astype(np.float)
    im = im / im.max()
    sy = snd.filters.sobel(im, axis=0, mode="constant")
    sx = snd.filters.sobel(im, axis=1, mode="constant")
    B1 = np.power(sy, 2)
    A1 = np.power(sx, 2)
    C1 = np.multiply(sx, sy)
    A = snd.filters.gaussian_filter(A1, 0.5)
    B = snd.filters.gaussian_filter(B1, 0.5)
    C = snd.filters.gaussian_filter(C1, 0.5)
    # C = C.astype(np.uint)
    k = 0.05
    corner_det = np.multiply(A, B) - np.power(C, 2)
    corner_trc = A + B
    cornerness = corner_det - k * np.power(corner_trc, 2)
    c_max = cornerness.max()
    c_thresh = 0.01 * c_max
    I_cornerness = cornerness > c_thresh
    # morphology - opening to separate objects better
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    I_cornerness_open = morphology.binary_dilation(I_cornerness, se, iterations=1)

    im_corner = im_ori
    im_corner[I_cornerness] = [255, 0, 0]

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(8, 8), sharex=True, sharey=True
    )

    ax1.imshow(cornerness, cmap=plt.cm.gray)
    ax1.axis("off")
    ax1.set_title("cornerness", fontsize=20)

    ax2.imshow(I_cornerness, cmap=plt.cm.gray)
    ax2.axis("off")
    ax2.set_title("I_cornerness", fontsize=20)

    ax3.imshow(im_corner, cmap=plt.cm.gray)
    ax3.axis("off")
    ax3.set_title("Image", fontsize=20)

    fig.tight_layout()

    plt.show()
