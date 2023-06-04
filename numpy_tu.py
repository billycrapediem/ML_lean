
import numpy as np
## create array

a = np.array([10,20,30,40])
b = np.arange(4)
x = np.arange(12).reshape((3,4))
print(x)
## basic operation
print(a,b)
d = a + b
c = a**2
print(c, d)

## boolean
print(c == 10)

# dot product
e = np.array([[1,1],[0,1]])
f = np.arange(4).reshape((2,2))


p_dot = np.dot(e,f)
print(p_dot)
# cross product


pp = e * f
print(pp)

##stat

st = np.random.random((2,4))
print(st)
print(np.sum(st),
np.min(st),
np.max(st))

# row and col op

m1 = np.arange(2,14).reshape((3,4))
print(m1)
print(np.argmin(m1),np.argmax(m1))
print(np.median(m1))
print(np.cumsum(m1))
m1[0][1] = 0
print(m1)
print(np.nonzero(m1))
print(np.sort(m1))

# index
m2 = np.arange(3,15).reshape((3,4))
print(m2)
print(m1[1,1:3])

for row in m2:
    print(row[1:3])

for col in m2.T:
    print(col)

## combine array
a1 = np.array([1,1,1])
a2 = np.array([2,2,2])


print(np.vstack((a1,a2))) # vertical stack
c1 = np.vstack((a1,a2))
c2 = np.hstack((a1,a2)) ## horizontal stack
print(c2)
print(c1.shape)

## combine array mutiple

## split
a3 = np.arange(12).reshape((3,4))

print(np.split(a3,2,axis=1))

## 不平均分割
print(np.vsplit(a3,3))
print(np.hsplit(a3,2))
print(a3)
print(a3[:,[1,2]])