from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math

##############################################
# 1D GAUSS FUNCTION
# sigma =20
# len = math.ceil(sigma*3)
# indices = np.arange(-len,len+1)
# gausMask1D= -1.*(indices)**2/(2*sigma**2)
# gausMask1D= 1/(math.sqrt(2*math.pi)*sigma)*np.exp(gausMask1D)
# sum =np.sum(gausMask1D, axis=None)
# gausMask1D = gausMask1D/sum

# plt.plot(gausMask1D)
# plt.show()

##############################################
# # 2D GAUSS FUNCTION
sigma =20
len = math.ceil(sigma*3)

x = np.arange(-len,len+1)
y = np.arange(-len,len+1)

gausMask2D = np.zeros((2*len+1,2*len+1))

for i in range(-len,len+1):
    for j in range(-len,len+1):
        constant = 1/(2*math.pi*sigma**2)
        exp_term = np.exp(-1*((x[i]**2+y[j]**2))/(2*sigma**2))
        gausMask2D[i,j]= constant*exp_term

sum =np.sum(gausMask2D, axis=None)
gausMask2D = gausMask2D/sum

# create grid to plot using numpy
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, gausMask2D, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

imgplot = plt.imshow(gausMask2D, cmap='gray')  
plt.show()

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


