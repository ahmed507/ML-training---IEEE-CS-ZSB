import numpy as np

# coefficients
a = np.array([[3, 2, 7, 1, 8],
              [1, 7, 8, 4, 1],
              [4, 8, 7, 2, 3],
              [1, 9, 8, 1, 3]])
# constants
b = np.array([5, 8, 1, 3])
# number of variables
n = len(a)
print("Before gaussian elimination")
print(np.append(a, b.reshape(n, 1), 1))
for i in range(0, n + 1):
    for j in range(i + 1, n):
        r = a[j, i] / a[i, i]
        for k in range(i, n):
            a[j, k] -= a[i, k] * r
            b[j] -= b[i] * r
print("after gaussian elimination")
print(np.append(a, b.reshape(n, 1), 1))
print("=========")
print("The Result : ", end="")
print(b)
