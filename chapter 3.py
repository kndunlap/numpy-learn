import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF

from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
import seaborn as sns

dir()

A = np.array([3, 5, 11])
dir(A)

A.sum()
A.median()
A.mean()

Boston = load_data("boston")
Boston.columns
Boston?

X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})

X
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
model
summarize(results)

plt.scatter(Boston['lstat'], Boston['medv'])

design = MS(['lstat'])
design = design.fit(Boston)
X = design.transform(Boston)
X[:4]
results.summary()
results.params


new_df = pd.DataFrame({'lstat': [5, 10, 15]})
newX = design.transform(new_df)
newX
new_predictions = results.get_prediction(newX)
new_predictions.predicted_mean


def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim)

ax = Boston.plot.scatter('lstat', 'medv')
abline(ax,
       results.params[0],
       results.params[1],
       'g--',
       linewidth=3)


X = MS(['lstat', 'age']).fit_transform(Boston)
X
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)

terms = Boston.columns.drop('medv')


X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

dir(results)

results.rsquared

vals = [VIF(X, i)
        for i in range(1, X.shape[1])]

vif = pd.DataFrame({'vif':vals},
                   index = X.columns[1:])

vif

X = MS(['lstat',
        'age',
        ('lstat', 'age')]).fit_transform(Boston)

model2 = sm.OLS(y, X)
summarize(model2.fit())

Carseats = load_data('Carseats')
Carseats.columns





#%% Tigers
from pybaseball import batting_stats
data = batting_stats(2023)

data2 = data.loc[:, ['BB', 'Barrel%', 'HardHit%', 'Z-Swing% (pi)', 'LD%', 'CSW%', 'Clutch']]
y = data['wRC+']

X = MS(data2).fit_transform(data)
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

y
X

sns.scatterplot(x = 'wRC+', y = 'Barrel%', data = data)

y = data['wRC+']
X = MS(['Barrel%',
        'Clutch',
('Barrel%', 'Clutch')]).fit_transform(data)
model2 = sm.OLS(y, X)
summarize(model2.fit())