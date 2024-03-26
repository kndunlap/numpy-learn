#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:37:10 2024

@author: u1314571
"""

import os
os.chdir('/Users/u1314571/Desktop')
os.chdir('Python_Organized')
os.chdir('pandas-learn')

#%% Intro 
import pandas as pd

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data[1]
data[1:3]


population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}

population = pd.Series(population_dict)
population
population['California':'Illinois']

pd.Series([2, 4, 6])
pd.Series(5, index=[100,200,300])

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
              'Florida': 170312, 'Illinois': 149995}

area_dict
area = pd.Series(area_dict)
area
population
area
states = pd.DataFrame({'population': population,
                       'area': area})
states.columns

data = [{'a': i, 'b':2 * i}
        for i in range(3)]
pd.DataFrame(data)

pd.DataFrame({'population': population,
              'area': area})

pd.DataFrame(np.random.rand(3, 2),
             columns = ['foo', 'bar'],
             index=['a', 'b', 'c'])

ind = pd.Index([2, 3, 5, 7, 11])
ind[1]
ind[::2]

#%% Data Indexing and Selection
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index = ['a', 'b', 'c', 'd'])
data['b']
'a' in data
data

data.loc[1:3]


area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data['density'] = data['pop']/data['area']
data.values
data.T
data.ix[:3, :'pop']
data.ix[:'New York', :'area']

data.loc[data.density < 100, ['pop', 'density']]
data[data.density > 100]

#%% Operating on Data in Pandas
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser
df = pd.DataFrame(rng.randint(0, 10, (3,4)),
                  columns = ['A', 'B', 'C', 'D'])
df
np.exp(ser)

np.sum(df + 1)

A = pd.DataFrame(rng.randint(0, 20, (2,2)),
                 columns = list('AB'))
A
B = pd.DataFrame(rng.randint(0, 10, (3,3)),
                 columns = list('BAC'))
B
A
B
A+B

A = rng.randint(10, size = (3,4))
A
A - A[0]

df = pd.DataFrame(A, columns=list('QRST'))
df
df - df.iloc[0]

df.subtract(df['R'], axis = 0)
halfrow = df.iloc[0, ::2]
halfrow
df
df - halfrow

#%% Handling Missing Data

vals1 = np.array([1, None, 3, 4])
vals1

vals2 = np.array([1, np.nan, 3, 4])
vals2.dtype

pd.Series([1, np.nan, 2, None])

data = pd.Series([1, np.nan, 'hello', None])
data
data.isnull()
data[data.notnull()]
data[data.isnull()]
data.dropna()

df = pd.DataFrame([[1, np.nan, 2],
                  [2, 3, 5],
                  [np.nan, 4, 6]])

df.dropna()
df.dropna(axis = 'columns')

data = pd.Series([1, np.nan, 2, None, 3], index = list('abcde'))
data
data.fillna(1)
data.fillna(method = 'ffill')

#%% Hierarchial Indexing










