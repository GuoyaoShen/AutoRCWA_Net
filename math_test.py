import numpy as np
import matplotlib.pyplot as plt
from skimage import draw

# a = np.array([[3,4],[5,6]])
# b = np.array([[1,0],[1,1]])
# print('a', a)
# print('b', b)
#
# a[b==1] = 7
#
# # print(b[:,0])
# # print(b[:,1])
# print(a)

# a = (5,6)
# print(tuple(ai//2 for ai in a))

img = np.zeros((100, 100), dtype=np.uint8)
rr, cc = draw.disk((0,0), 50, shape=img.shape)
img[rr, cc] = 1
# print(img)
plt.figure(1)
plt.imshow(img * 255, cmap='gray')
plt.show()