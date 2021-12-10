from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import scipy.ndimage as snd

##############################################
# # 2D LAPLACIAN FUNCTION
im = np.array(
    Image.open("BM5113 - Bilgisayarla Görme/images/jpg/car1.jpg").convert("L")
)

sigma = 3
len = math.ceil(sigma * 3)

x = np.arange(-len, len + 1)
y = np.arange(-len, len + 1)

laplacianMask2D = np.zeros((2 * len + 1, 2 * len + 1))

for i in range(-len, len + 1):
    for j in range(-len, len + 1):
        constant = (
            -1
            / (math.pi * sigma ** 4)
            * (1 - (x[i] ** 2 + y[j] ** 2) / (2 * sigma ** 2))
        )
        exp_term = np.exp(-1 * ((x[i] ** 2 + y[j] ** 2)) / (2 * sigma ** 2))
        laplacianMask2D[i, j] = constant * exp_term

im = im.astype(np.float)
im_lap1 = snd.convolve(im, laplacianMask2D, mode="constant")
print(np.sum(laplacianMask2D))
sum = np.sum(laplacianMask2D, axis=None)
laplacianMask2D = laplacianMask2D / sum
print(np.sum(laplacianMask2D, axis=None))
im_lap2 = snd.convolve(im, laplacianMask2D, mode="constant")
maxel = np.abs(im_lap2).max()
edges = np.abs(im_lap2) > 0.1 * maxel
# im_list2 = np.concatenate((im, im_lap1, im_lap2), axis=1)
imgplot = plt.imshow(edges, cmap="gray")
plt.show()


imgplot = plt.imshow(laplacianMask2D)
plt.show()
# create grid to plot using numpy
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.gca(projection="3d")

# Plot the surface.
surf = ax.plot_surface(
    X, Y, laplacianMask2D, cmap=cm.coolwarm, linewidth=0, antialiased=False
)

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
