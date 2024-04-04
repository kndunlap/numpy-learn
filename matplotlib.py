#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:04:42 2024

@author: u1314571
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline

x = np.linspace(0, 10, 100)
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.sin(x), '--')

#%% Simple Line Plots
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), color = 'y', linestyle = 'dotted')

fig = plt.figure()
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set_xlim(-1, 11)
ax.set_ylim(-1.5, 1.5)
plt.show()

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 100)
ax.set_xlim(-1, 11)
ax.set_ylim(-1.5, 1.5)
ax.plot(x, np.sin(x), color = 'red')
ax.plot(x, np.cos(x), color = 'blue')

#%% Simple Scatter Plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import numpy as np
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color = 'black')


plt.scatter(x, y, marker = 'o')

rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c = colors, s = sizes, alpha = 0.3, cmap = 'viridis')
plt.colorbar()


#%% Contours and Density Plots
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
import numpy as np
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, 20, cmap = 'RdGy')
plt.colorbar()

#%% Histograms

data = np.random.randn(1000)
plt.hist(data)

counts, bin_edges = np.histogram(data, bins = 20)
print(counts)

#%% Customizing Plot Legends
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label = 'Sine')
ax.plot(x, np.cos(x), '--r', label = 'Cosine')
ax.axis('equal')
leg = ax.legend()

y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)
plt.legend(lines[:2], ['first', 'second'])

#%% Customizing colorbars

x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
x
I
plt.imshow(I)
plt.colorbar();

plt.imshow(I, cmap = 'gray')

#%% Visualization with Seaborn

rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.rand(500, 6), 0)
plt.plot(x, y)
plt.legend('ABCDEF', ncol = 2, loc = 'upper left')

import seaborn as sns
sns.set()
plt.plot(x, y)
plt.legend('ABCDEF', ncol = 2, loc = 'upper left')
