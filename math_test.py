import numpy as np

# a = [1,2,3,7,21,13]
# print(max(a))

a = np.array([[2,4],[2,3],[1,7]])
b = np.array([1,7])

# print(np.isin(b,a))
# print(a-b)
# # print(np.sum(a-b, axis=-1))
# # print(np.sum(a-b, axis=-1)==0)
# # print(np.any(np.sum(a-b, axis=-1)==0))
#
# print(a-b==np.array([0,0]))
# print(np.any(np.all(a-b==np.array([0,0]), axis=-1)))

print(np.concatenate((a,b[np.newaxis,...]), axis=0))

list = []
print(list==[])

print(np.around(13.777,0))