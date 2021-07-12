import os, sys
import pandas as pd
import time, datetime

df = pd.read_csv('../LoadTesting Tool/res.txt')
df = df[['timeStamp','elapsed', 'label', 'responseCode']]
df['dt'] =  pd.to_datetime(df['timeStamp'], unit='ms').dt.floor('s')
del df['timeStamp']
df

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

tdf = df.groupby(['dt', 'label'])['elapsed'].agg(['mean', 'count', percentile(75), percentile(90)]).reset_index()
tdf.to_csv('res.csv')


import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return c*x**b #+ b*x + c

def func2(x, a, b, c):
    return a * np.exp(-b * x) + c

def func3(x, a, b, c):
    return b**c

ttdf = tdf[tdf['percentile_90'] < np.percentile(tdf['percentile_90'], 90)]
f = tdf[tdf['percentile_90'] > np.percentile(tdf['percentile_90'], 90)]

fx = f['count']
fy = f['percentile_90']

x = ttdf['count']
# y = tdf['percentile_75']
y = ttdf['percentile_90']

yn = y #y + 0.1*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, y)

plt.figure(figsize=(16, 10), dpi=160)
plt.plot(x, yn, 'ko', label="Original Noised Data", markersize=1)
plt.plot(x, func(x, *popt), 'r', label="Fitted Curve")

plt.plot(fx, fy, 'ro', label="filtred", markersize=1)
plt.legend()
plt.show()