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

plt.plot(x, y, "o")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
