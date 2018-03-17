import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

a = np.ones([7,7,3])
a[1:-1,1:-1,:]=0
print(a)
b = scipy.misc.imresize(a[:,:,0],[84,84,1], interp='nearest')
c = scipy.misc.imresize(a[:,:,1],[84,84,1], interp='nearest')
d = scipy.misc.imresize(a[:,:,2],[84,84,1], interp='nearest')
a = np.stack([b,c,d],axis=2)
plt.imshow(a, interpolation="nearest")
plt.show()