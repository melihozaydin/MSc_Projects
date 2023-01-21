import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_img(img_array, title="Image", cmap=None):
    
    if (cmap is None) and ((len(img_array.shape) < 2) or (np.array(img_array.shape) == 1).any()):
        img_array = np.squeeze(img_array)
        cmap = "gray"

    plt.title(title, fontsize=12)
    plt.suptitle(
            f"shape: {img_array.shape}, dtype: {img_array.dtype}\n" 
            f"range: {np.min(img_array)} - {np.max(img_array)}\n"
            f"mean: {np.mean(img_array):.2f}, std: {np.std(img_array):.2f}", 
            fontsize=8)

    plt.imshow(img_array, cmap=cmap)
    plt.tight_layout()
    plt.show()

def show_hist(img, channel_order="RGB", title="Histogram", show_image=True, ret=False, bins=256, cumulative=False):
    # plot histograms for all 3 channels
    #print("len(img.shape):",len(img.shape))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    
    # GRAYSCALE
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        color_names = ["Gray"]
    else:
        # RGB
        color_names = ["Red", "Green", "Blue"]
        if channel_order == "BGR":
            # BGR
            color_names = reversed(color_names)
    if show_image:
        show_img(img, title="Image")

    if not cumulative:
        fig, axes = plt.subplots(1, img.shape[2], figsize=(15,5))
        if img.shape[2] == 1:
            axes = [axes]
            fig.set_size_inches(7,5)

        fig.tight_layout()

        for i, ch_name, ax in zip(range(img.shape[2]), color_names, axes):
            ax.hist(img[:, :, i].ravel(), bins=bins, range=[0, 256], )
            ax.set_title(f"{title}-{ch_name}")
    else:
            fig, ax = plt.subplots(1, 1, figsize=(7,5))
            ax.hist(img.ravel(), bins=bins, range=[0, 256], cumulative=True, )
            ax.set_title(f"{title}-Cumulative")

    plt.show()
    plt.close()
    if ret:
        return fig

def img_read(src, ret_bgr=False, ret_gray=False, show=True):
    """
    Reads an image from the path and returns it.
    """
    img = cv2.imread(src) # BGR
    if img is None:
        raise FileNotFoundError(f"Could not find image file '{src}'.")
    if ret_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # bgr 2 gray
    elif not ret_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr 2 rgb

    if show:
        show_img(img, title="Image")
    
    return img

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians


def find_poi(image, sigma_min, sigma_max, threshold, visualize=False):
    # Create the scale space by applying DoG filtering
    scale_space = difference_of_gaussians(image, sigma_min, sigma_max)
    # Find local maxima
    local_max = peak_local_max(scale_space, threshold_rel=threshold, min_distance=1, indices=True)
    # Find local minima by reversing the sign of the scale space
    scale_space = -scale_space
    local_min = peak_local_max(scale_space, threshold_rel=threshold, min_distance=1, indices=True)

    hessian = np.transpose(np.array(hessian_matrix(image, sigma=sigma_max, order='rc')), (1,2,0))
    hessian_th = hessian.max() * threshold
    print(hessian.shape)
    print("hessian_th: ", hessian_th)

    # Use the Hessian matrix to clean up the points
    final_points = []
    """
    for point in np.concatenate((local_max, local_min)):
        eig_val, eig_vec = np.linalg.eig(hessian[point[0]][point[1]])
        if eig_val[0] < threshold and eig_val[1] < threshold:
            continue
        # Append to the points of interest
        points_of_interest.append(point)
    """

    print("np.concatenate((local_max, local_min)).shape", np.concatenate((local_max, local_min)).shape)
    for point in np.concatenate((local_max, local_min)):
        if (hessian[point[0]][point[1]] > hessian_th).all():
            final_points.append(point)

    final_points = np.array(final_points)
    print("final points_of_interest .shape: \n", final_points.shape)

    if visualize:
        # Plot the image
        plt.imshow(image, cmap='gray')
        # Plot the points of interest on top of the image
        plt.plot(final_points[:, 1], final_points[:, 0], 'ro', alpha=0.1)
        plt.show()

    return final_points


# Global Histogram Equalization
# Histogram Equalization
#https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43
def hist_equalization(img_array):

    # STEP 1: Normalized cumulative histogram
    #flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)
    #normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels
    #cumulative histogram
    chistogram_array = np.cumsum(histogram_array)

    # STEP 2: Pixel mapping lookup table
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)


    # STEP 3: Transformation
    eq_img_array = np.take(transform_map, img_array)
    return eq_img_array

def log_transform(img_array, c=1):

    # Apply log transformation method
    c = 255 / np.log(1 + np.max(img_array))
    log_image = c * (np.log(img_array + 1))
    log_image = np.array(log_image, dtype = np.uint8)

    return log_image

def add_noise(img_array, noise_typ):
    """
    Parameters
    ----------
    img_array : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row,col,ch= img_array.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img_array + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = img_array.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img_array)
        # Salt mode
        num_salt = np.ceil(amount * img_array.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in img_array.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img_array.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in img_array.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(img_array))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img_array * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = img_array.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = img_array + img_array * gauss
        return noisy

def add_gauss_noise(img, mean=0, std_dev=None, noise_mult=1):
    var = np.var(img)
    sigma = std_dev or var**0.5  # std_dev

    noise = np.random.randint(mean, sigma, img.shape)
    noisy = img + noise*noise_mult
    noisy = np.array(np.clip(noisy, 0, 255), dtype=np.uint8)
    return noisy

def apply_gauss_filter(img, kernel_size=5, sigma=0):
    kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    kernel_size = (kernel_size[0] if kernel_size[0]%2 != 0 else kernel_size[0]+1, 
                    kernel_size[1] if kernel_size[1]%2 != 0 else kernel_size[1]+1)
    print()
    print("sigma:", sigma)
    # apply gaussian filter
    # https://docs.opencv.org/4.5.2/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    img_blur = cv2.GaussianBlur(img, ksize=kernel_size, sigmaX=sigma)
    show_img(img_blur, title=f"sigma: {sigma}, kernel_size: {kernel_size}")
    return img_blur