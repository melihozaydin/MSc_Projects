from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


def gaussianFilter(im, sigma=1):
    halfwid = 3 * sigma
    num = 2 * halfwid + 1
    num = 7
    xx = np.linspace(-halfwid, halfwid, num)
    yy = np.linspace(-halfwid, halfwid, num)
    xx, yy = np.meshgrid(xx, yy)

    e = -1 / (np.pi * (sigma ** 4))
    d = 1 - (((xx * 2) + (yy * 2)) / (2 * (sigma ** 2)))

    a = 1 / (2 * np.pi * (sigma ** 2))
    exp = np.e * (-((xx * 2) + (yy * 2)) / (2 * (sigma ** 2)))
    g = a * exp
    l = e * d * exp

    # laplace = convo(im, l)
    gauss = convo(im, g)

    return gauss


def gradients(im):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # kernel_x = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
    # kernel_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1,0]])

    sx = convo(im, kernel_x)
    sy = convo(im, kernel_y)
    return sx, sy


def convo(image, filter):

    # setting parameters
    image_row, image_col = image.shape
    filter_row, filter_col = filter.shape

    # applies padding(greater than size of original image)
    # and return output image(size of original image)
    padded_image, output = padding(image, filter)

    # applies filter pixelwise
    for row in range(image_row):
        for col in range(image_col):

            # applies filters
            output[row, col] = np.sum(
                filter * padded_image[row : row + filter_row, col : col + filter_col]
            )

    return output


def padding(image, filter):

    # setting parameters
    image_row, image_col = image.shape
    filter_row, filter_col = filter.shape

    # generating new output image
    output = np.zeros((image_row, image_col))

    # setting pad size
    pad_height = int((filter_row - 1) / 2)
    pad_width = int((filter_col - 1) / 2)

    # applies padding to image
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[
        pad_height : padded_image.shape[0] - pad_height,
        pad_width : padded_image.shape[1] - pad_width,
    ] = image

    return padded_image, output


def harrisCorner(inputArr):

    inputt = inputArr

    # bluring for noise reduction
    # inputArr = gaussianFilter(inputArr,1)

    # get derivatives using sobel kernels
    sx, sy = gradients(inputArr)

    B1 = np.power(sy, 2)
    A1 = np.power(sx, 2)
    C1 = np.multiply(sx, sy)

    # gaussian filter kernel = 3*2*sigma+1 = 7==> 7x7
    A = gaussianFilter(A1, sigma=1)
    B = gaussianFilter(B1, sigma=1)
    C = gaussianFilter(C1, sigma=1)

    k = 0.05

    corner_det = np.multiply(A, B) - np.power(C, 2)
    corner_trc = A + B
    cornerness = corner_det - k * np.power(corner_trc, 2)

    c_max = cornerness.max()
    c_thresh = 0.001 * c_max
    I_cornerness = cornerness > c_thresh
    im_corner = np.zeros((inputArr.shape[0], inputArr.shape[1], 3), dtype="uint8")

    im_corner[:, :, 0] = inputt
    im_corner[:, :, 1] = inputt
    im_corner[:, :, 2] = inputt

    im_corner[I_cornerness] = [255, 0, 0]

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(8, 8), sharex=True, sharey=True
    )

    ax1.imshow(I_cornerness, cmap=plt.cm.gray)
    ax1.axis("off")
    ax1.set_title("I_cornerness", fontsize=20)

    ax2.imshow(im_corner, cmap=plt.cm.gray)
    ax2.axis("off")
    ax2.set_title("Image", fontsize=20)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    import time    
    img = cv2.imread("BM5113 - Bilgisayarla Görme/images/building2-2.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (640, 640))
    im = np.array(img, dtype=float)
    start = time.time()
    harrisCorner(im)
    print("Time taken: ", time.time() - start)
