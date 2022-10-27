from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.ndimage as snd

im = np.array(Image.open('./images/jpg/trailer.jpg').convert('L'))

fig, ax = plt.subplots(nrows=1, ncols=4)
plt.gray()

for sig in range(1,11,3): 
    out = snd.filters.gaussian_filter(im,sig)
    indexAxis = math.floor((sig-1)/3)+1
    plt.subplot(1,4,indexAxis), plt.imshow(out)
    plt.title((r'Gauss blur, $\sigma=$'+str(sig)))

plt.show()