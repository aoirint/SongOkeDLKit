import numpy as np

a = np.array([ 0, 1, 0, 1, 2, 3, 2, 3 ])
print(a.shape)
print(a)

b = a.reshape((-1, 2))
print(b.shape)
print(b)

c = b.transpose((1, 0))
print(c.shape)
print(c)
