import numpy as np

a = np.array([[3,4],[5,6]])
b = np.array([[1,0],[1,1]])
print('a', a)
print('b', b)

a[b==1] = 7

# print(b[:,0])
# print(b[:,1])
print(a)