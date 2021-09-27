import matplotlib.pyplot as plt
import csv

x = []
y = []
with open('dataset.csv') as file:
    reader = csv.reader(file)
    for i, j in reader:
        if str.isdecimal(i):
            x.append(float(i))
            y.append(float(j))

m = len(x)
alpha = 0.0005
theta = [1,1]
h = [None]*m
y_res = []
for zzz in range(10000):
    for i in range(m):
        h[i] = theta[0] + theta[1] * x[i]
        theta[0] = theta[0] - ((alpha / m) * (h[i] - y[i]))
        theta[1] = theta[1] - (((alpha / m) * (h[i] - y[i]))*x[i])
for j in range(len(x)):
    hyp = theta[0] + theta[1] * x[j]
    y_res.append(hyp)
# print(theta)
plt.plot(x, y, "o")
plt.plot(x,y_res)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
