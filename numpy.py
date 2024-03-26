#%% Understanding Data Types in Python
L = list(range(10))
L
L3 = [True, "2", 3.0, 4]
L3
type(L3)
[type(item) for item in L3]
import array
L = list(range(10))
A = array.array('i', L)
A

import numpy as np

np.array([1, 4, 2, 5, 3])

np.array([1,2,3,4], dtype = 'float32')

np.ones((3, 4), dtype=int)

#%% Basics of numpy arrays
import numpy as np
np.random.seed(0)
x1 = np.random.randint(10, size = 6)
x2 = np.random.randint(10, size = (3, 4))
x3 = np.random.randint(10, size = (3, 4, 5))
x1
x2
x3

print("dtype:", x3.dtype)

x = np.arange(10)
x[:5]
x[5:]
x[4:7]

x2[1, :]
x2
print(x2)
x2_sub = x2[:2, :2]
print(x2_sub)
x2_sub[0, 0] = 99
print(x2_sub)
print(x2)

grid = np.arange(1, 10).reshape((3,3))
print(grid)

x = np.array([1, 2, 3])
x.reshape((1,3))
x[np.newaxis, :]
x.reshape

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
x1
x2
x3

#%% Computation on Numpy Arrays
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)



big_array = np.random.randint(1, 100, size=1000000)
%timeit compute_reciprocals(big_array)

print(compute_reciprocals(values))


%timeit compute_reciprocals(big_array)


print(1.0/values)


%timeit (1.0 / big_array)
np.arange(5) / np.arange(1,6)

x = np.arange(9).reshape((3,3))
2 ** x

x = np.arange(4)
x
print("x + 5 =", x + 5)
print("x ** 19 = ", x ** 19)
3**19

np.add(x, 2)
np.power(x, 19)

theta = np.linspace(0, np.pi, 3)
theta

print("sin(theta) =", np.sin(theta))

x = np.arange(1, 6)
np.add.reduce(x)

np.add.outer(x, x)

#%% Aggregation
import numpy as np
L = np.random.random(100)
sum(L)
np.sum(L)

M = np.random.random((3,4))
print(M)
M.sum()
M.min()
M.min(axis = 0)
M.min(axis = 1)


!head -4 president_heights.csv
import pandas as pd
data = pd.read_csv('president_heights.csv')
heights = np.array(data['height(cm)'])
heights
data

print("Mean height:", heights.mean())
np.percentile(heights, 25)
np.median(heights)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.hist(heights)

#%% Computation on arrays - broadcasting.

import numpy as np
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

M = np.ones((2,3))
a = np.arange(3)
M + a

a = np.arange(3).reshape((3,1))
a
b = np.arange(3)
a
b
a.shape = (3,1)
b.shape = (3,)
a.shape
b.shape
a
b

M = np.ones((3,2))
a = np.arange(3)
M
a
M + a

a[:, np.newaxis].shape
M + a[:, np.newaxis]
a

X = np.random.random((10,3))
X
Xmean = X.mean(0)
Xmean
X_centered = X - Xmean
X_centered
X_centered.mean(0)

#%% Comparisons, Masks and Boolean Logic

import pandas as pd
rainfall = pd.read_csv("Seattle2014.csv")['PRCP'].values
inches = rainfall/254.0
inches.shape
plt.hist(inches, 40)

x = np.array([1, 2, 3, 4, 5])
x < 3
x > 3
x <= 3

(2 * x) == (x ** 2)


rng = np.random.RandomState(0)
x = rng.randint(10, size = (3, 4))
x
print(x)

np.count_nonzero(x < 6)

np.any(x == 8)

np.sum((inches > 0.5) & (inches < 1))

np.sum(~( (inches <=0.5) | (inches >=1)))

print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
                                        
x
x = x[x < 5]
x

rainy = (inches > 0)
days = np.arange(365)
days
summer = (days > 172) & (days < 262)
np.max(inches[summer])
rainy
inches
summer

#%% Fancy Indexing

rand = np.random.RandomState(42)
x = rand.randint(100, size = 10)
x
[x[3], x[7], x[2]]
ind = [3, 7, 4]
x[ind]

ind = np.array([[3, 7],
                [4, 5]])

x[ind]

X = np.arange(12).reshape((3, 4))
X
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
X[row, col]

X[row[:, np.newaxis], col]
X

row[:, np.newaxis] * col

print(X)
X[2, [2, 0, 1]]

X[1:, [2, 0, 1]]

mask = np.array([1, 0, 1, 0], dtype = bool)
mask
X[row[:, np.newaxis], mask]

mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:, 0], X[:, 1]);

indices = np.random.choice(X.shape[0], 20, replace = False)
indices
selection = X[indices]
selection.shape

x = np.arange(10)
i = np.array([2, 1, 8, 4])
i
x[i] = 99
x
i



x = np.zeros(10)
x
x[[0, 0]] = [4, 6]
x

i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)

np.random.seed(42)
x = np.random.randn(100)
x
bins = np.linspace(-5, 5, 20)
bins
counts = np.zeros_like(bins)
counts
i = np.searchsorted(bins, x)
i
np.add.at(counts, i , 1)
bins

counts
plt.plot(bins, counts)

#%% Sorting Arrays

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.array([2, 1, 4, 3, 5])
selection_sort(x)

x = np.array([2, 1, 4, 3, 5])
np.sort(x)
x.sort()
x
print(x)

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)

x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)

X = rand.rand(10, 2)
X

plt.scatter(X[:, 0], X[:, 1], s = 100)

dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
dist_sq

dist_sq.diagonal()
nearest = np.argsort(dist_sq, axis = 1)
print(nearest)
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis = 1)
nearest_partition

differences = X[:, np.newaxis, :] - X[np.newaxis, :, ]
differences

#%% Structured Data
name = ['Alice', 'Bob', 'Cathy', 'Doug']
name
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype = int)
x
data = np.zeros(4, dtype = {'names':('name', 'age', 'weight'),
                            'formats':('U10', 'i4', 'f8')})
data
data['name'] = name
data['age'] = age
data['weight'] = weight
data
data['name']
data[0]
data[-1]['name']
data[data['age'] < 30]['name']

np.dtype([('name', 'S10'), ('age', '<i4'), ('weight', '<f8')])

import os
os.chdir('/Users/u1314571/Desktop')
os.getcwd()











