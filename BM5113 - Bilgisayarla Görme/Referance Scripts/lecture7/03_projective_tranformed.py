import numpy as np
import matplotlib.pyplot as plt
import cv2

# With jupyter notebook uncomment below line
# %matplotlib inline
# This plots figures inside the notebook


def plot_cv_img(input_image):
    """
    Converts an image from BGR to RGB and plots
    """
    # change color channels order for matplotlib
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    # For easier view, turn off axis around image
    plt.axis("off")
    plt.show()


def main():
    # read an image
    img = cv2.imread("./images/jpg/car1.jpg")
    # img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # generate transformation matrix form preselected points
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    perpective_tr = cv2.getPerspectiveTransform(pts1, pts2)

    transformed = cv2.warpPerspective(
        img,
        perpective_tr,
        (np.int32(img.shape[1] * 2 / 3), np.int32(img.shape[0] * 4 / 3)),
    )

    # Do plot
    plot_cv_img(transformed)
    img = cv2.imread("./images/jpg/football.jpg")
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [75, 80]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250], [120, 300]])
    affine_tr = cv2.getPerspectiveTransform(pts1, pts2)

    transformed = cv2.warpPerspective(
        img, affine_tr, (img.shape[1] * 2, img.shape[0] * 2)
    )
    plot_cv_img(transformed)


if __name__ == "__main__":
    main()
