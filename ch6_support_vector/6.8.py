import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm

from __init__ import *


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
data = pd.read_csv(data_path)

X = data.iloc[:]['密度'].values.reshape(-1, 1)
y = data.iloc[:]['含糖率'].values

gamma = 10
C = 1

ax = plt.subplot()
set_ax_gray(ax)
ax.scatter(X, y, color='c', label='data')

for gamma in [1, 10, 100, 1000]:
    svr = svm.SVR(kernel='rbf', gamma=gamma, C=C)  # gamma为核系数
    svr.fit(X, y)

    ax.plot(np.linspace(0.2, 0.8), svr.predict(np.linspace(0.2, 0.8).reshape(-1, 1)),
            label='gamma={}, C={}'.format(gamma, C))

ax.legend(loc='upper left')
ax.set_xlabel('密度')
ax.set_ylabel('含糖率')
plt.show()
