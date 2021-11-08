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

def img_read(src, ret_bgr=False, ret_gray=False):
    """
    Reads an image from the path and returns it.
    """
    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError(f"Could not find image file '{src}'.")
    if not ret_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr 2 rgb
    elif ret_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # bgr 2 gray
        
    plt.imshow(img)
    return img