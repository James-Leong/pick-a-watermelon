import os
from collections import defaultdict

import numpy as np
import pandas as pd

from __init__ import *


def normal_distrubution_density(x, u, sigma):
    """正态分布的密度函数"""
    return 1 / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - u)**2 / (2*sigma**2))


class OddsDict(dict):
    def __init__(self, u, sigma):
        self.u = u
        self.sigma = sigma
    
    def __getitem__(self, key):
        return normal_distrubution_density(key, self.u, self.sigma)


class LaplaceNaiveBayes:
    def __init__(self, data, label, discrete_attr, continuous_attr):
        self.label = label
        self.discrete_attr = discrete_attr
        self.continuous_attr = continuous_attr
        self.attr = list(set(self.discrete_attr or []) | set(self.continuous_attr or []))
        self.y = data[label].unique()
        N = len(self.y)
        N_i = {a: data[a].nunique() for a in self.discrete_attr}  # 只有离散属性需要拉普拉斯修正
        D = len(data)
        Dc = {c: len(data[data[label] == c]) for c in self.y}
        # 采用拉普拉斯修正的类先验概率
        self.p_c = {c: (Dc[c] + 1) / (D + N) for c in self.y}
        # 采用拉普拉斯修正的属性条件概率
        self.p_xi_c = defaultdict(dict)
        for c in self.y:
            for a in self.discrete_attr:
                for xi in data[a].unique():
                    _index = np.logical_and(data[a] == xi, data[label] == c)
                    self.p_xi_c[c][f'{a}_{xi}'] = (len(data[_index]) + 1) / (Dc[c] + N_i[a])
            for a in self.continuous_attr:
                arg = data[data[label] == c][a].mean()
                sgm = data[data[label] == c][a].std()
                self.p_xi_c[c][a] = OddsDict(u=arg, sigma=sgm)

    def predict(self, sample):
        argmax = None, 0
        for c in self.y:
            print('------------------------------')
            print(f'odds of label {c}: {self.p_c[c]}', end='')
            odds = self.p_c[c]
            for a in self.attr:
                _xi_odds = self.p_xi_c[c][f'{a}_{sample[a]}'] if a in self.discrete_attr else self.p_xi_c[c][a][sample[a]]
                print(f' × {_xi_odds}', end='')
                odds *= _xi_odds
            print(f' = {odds}')
            if odds > argmax[1]:
                argmax = c, odds
        print('------------------------------')
        print(f'choose label: {argmax[0]}, odds: {argmax[1]}')
        print('==============================')
        return argmax


if __name__ == '__main__':
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    data = pd.read_csv(data_path)
    data = data.drop(columns='编号')
    # 类别标记
    label = '好瓜'
    # 属性集
    attr = [a for a in data.columns if a != label]
    continuous_attr = ['密度', '含糖率']
    discrete_attr = [a for a in attr if a not in continuous_attr]

    model = LaplaceNaiveBayes(data, label, discrete_attr, continuous_attr)

    sample = data.iloc[0]
    print('sample 1: \n', sample)
    model.predict(sample)
