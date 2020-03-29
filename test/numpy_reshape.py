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

d = c.transpose((1, 0))
print(d.shape)
print(d)

e = d.reshape((-1, ))
print(e.shape)
print(e)
