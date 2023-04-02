import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


def plt_support_(model, X_, y_, kernel, c):
    """
    绘制支持向量

    Args:
        clf (_type_): _description_
        X_ (_type_): _description_
        y_ (_type_): _description_
        kernel (_type_): _description_
        c (_type_): _description_
    """
    pos = y_ == 1
    neg = y_ == -1
    ax = plt.subplot()

    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(0, 0.8, 600)
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    Z_rbf = model.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})

    set_ax_gray(ax)

    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    ax.scatter(X_[model.support_, 0], X_[model.support_, 1], marker='o', c='silver', edgecolors='g', s=150,
               label='support_vectors')
    ax.legend()
    ax.set_title('{} kernel, C={}'.format(kernel, c))
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.close('all')

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    data = pd.read_csv(data_path)
    X = data.iloc[:][['密度', '含糖率']].values
    y = data.iloc[:]['好瓜'].values
    y[y == '是'] = 1
    y[y == '否'] = -1
    y = y.astype(int)

    C = 1000 # 正则化系数

    # 线性核
    svm_linear = svm.SVC(C=C, kernel='linear')
    svm_linear.fit(X, y)
    print('线性核：')
    print('预测值：', svm_linear.predict(X))
    print('真实值：', y)
    print('支持向量：', svm_linear.support_)

    # 高斯核
    svm_rbf = svm.SVC(C=C, kernel='rbf')
    svm_rbf.fit(X, y)
    print('高斯核：')
    print('预测值：', svm_rbf.predict(X))
    print('真实值：', y)
    print('支持向量：', svm_rbf.support_)

    plt_support_(svm_linear, X, y, 'linear', C)
    plt_support_(svm_rbf, X, y, 'rbf', C)
