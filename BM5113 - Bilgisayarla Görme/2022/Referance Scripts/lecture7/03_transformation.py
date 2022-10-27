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
    # img = cv2.imread('./images/jpg/football.jpg')
    # # generate transformation matrix
    # translation_matrix = np.float32([[1,0,160],[0,1,40]])
    # transformed = cv2.warpAffine(img, translation_matrix, (img.shape[1]*2,img.shape[0]*2))
    # # Do plot
    # plot_cv_img(transformed)
    # sx = 0.25
    # sy = 0.75

    img = cv2.imread("./images/png/kobi.png")
    translation_matrix = np.float32([[1, 0, 500], [0, 1, 250], [0, 0, 1]])
    shear_matrix = np.float32([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]])
    scale_matrix = np.float32([[0.25, 0, 0], [0, 0.75, 0], [0, 0, 1]])
    rotation_matrix = np.float32(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0],
            [np.sin(np.pi / 3), np.cos(np.pi / 3), 0],
            [0, 0, 1],
        ]
    )
    combined_matrix = np.matmul(scale_matrix, rotation_matrix)
    combined_matrix = np.matmul(shear_matrix, combined_matrix)
    combined_matrix = np.matmul(translation_matrix, combined_matrix)
    translation_matrix = np.float32([[1, 0, 100], [0, 1, -200], [0, 0, 1]])
    combined_matrix = np.matmul(translation_matrix, combined_matrix)
    combined_matrix = combined_matrix[0:2, :]
    transformed = cv2.warpAffine(
        img,
        combined_matrix,
        (np.int32(img.shape[1] * 3 / 4), np.int32(img.shape[0] * 4 / 3)),
    )
    # Do plot
    plot_cv_img(transformed)

    # translation_matrix = np.float32([[0,1,160],[1,0,40],[0.002,0.001,0.5]])
    # transformed = cv2.warpPerspective(img, translation_matrix, (img.shape[1]*2,img.shape[0]*2))

    # # Do plot
    # plot_cv_img(transformed)


if __name__ == "__main__":
    main()
