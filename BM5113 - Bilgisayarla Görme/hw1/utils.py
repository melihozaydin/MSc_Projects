import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_hist(img, channel_order="RGB", title="Histogram", ret=False, 
                      bins=256, cumulative=False):
    # plot histograms for all 3 channels
    #print("len(img.shape):",len(img.shape))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    
    # GRAYSCALE
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        color_names = ["Gray"]
    else:
        # RGB        
        color_names = ["Red", "Green","Blue"]
        if channel_order == "BGR":
            # BGR
            color_names = reversed(color_names)    

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
        if ret_gray:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(f"shape: {img.shape}")
        plt.show()
    return img


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

