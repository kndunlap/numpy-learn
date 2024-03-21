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

