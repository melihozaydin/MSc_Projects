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
im = np.array(Image.open('./images/jpg/car1.jpg').convert('L'))

sigma =3
len = math.ceil(sigma*5)

x = np.arange(-len,len+1)
y = np.arange(-len,len+1)
# create grid to plot using numpy
X, Y = np.meshgrid(x, y)
#\ line continuation character in Python
laplacianMask2Dm = (-1/(math.pi*sigma**4))*(1-((X**2+Y**2)/(2*sigma**2)))\
    *np.exp(-1*((X**2+Y**2))/(2*sigma**2))

print(np.sum(laplacianMask2Dm, axis=None))

# laplacianMask2D = np.zeros((2*len+1,2*len+1))
# for i in range(-len,len+1):
#     for j in range(-len,len+1):
#         constant = -1/(math.pi*sigma**4)*(1-(x[i]**2+y[j]**2)/(2*sigma**2))
#         exp_term = np.exp(-1*((x[i]**2+y[j]**2))/(2*sigma**2))
#         laplacianMask2D[i,j]= constant*exp_term
# im = im.astype(np.float)
# print(np.sum(laplacianMask2D))

imgplot = plt.imshow(laplacianMask2Dm) 
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, laplacianMask2Dm, cmap=cm.coolwarm,linewidth=0,\
     antialiased=False)
plt.show()
im= np.float64(im)
# im_lap1 = snd.convolve(im, laplacianMask2D, mode='constant')
im_lap2 = snd.convolve(im, laplacianMask2Dm, mode='constant')
maxel=np.abs(im_lap2).max()
edges = np.uint8(np.abs(im_lap2)>0.1*maxel)*255
print(edges.max())

im_list = np.concatenate((im, im_lap2), axis=1)

imgplot = plt.imshow(edges, cmap='gray') 
plt.show()






