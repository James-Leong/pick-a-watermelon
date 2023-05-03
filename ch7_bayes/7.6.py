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


class AODEBayes:
    def __init__(self, data, label, discrete_attr, continuous_attr):
        self.label = label
        self.discrete_attr = discrete_attr
        self.continuous_attr = continuous_attr
        self.attr = list(set(self.discrete_attr or []) | set(self.continuous_attr or []))
        self.y = data[label].unique()
        N = len(self.y)
        # 第i个属性可能的取值数
        N_i = {a: data[a].nunique() for a in self.attr}
        # 总的样本数量
        D = len(data)
        # 类别为c且在第i个属性上取值为xi的样本集合
        Dc_xi = defaultdict(dict)
        for c in self.y:
            for a in self.attr:
                for xi in data[a].unique():
                    Dc_xi[c][f'{a}_{xi}'] = len(data[np.logical_and(data[label]==c, data[a]==xi)])
        # 采用拉普拉斯修正的父属性与类别的先验联合概率
        self.p_c_xi = defaultdict(dict)
        for c in self.y:
            for a in self.attr:
                for xi in data[a].unique():
                    self.p_c_xi[c][f'{a}_{xi}'] = (Dc_xi[c][f'{a}_{xi}'] + 1) / (D + N_i[a] * len(self.attr))
        # 采用拉普拉斯修正的有独依赖的属性条件概率
        self.p_xj_c_xi = defaultdict(lambda: defaultdict(dict))
        for c in self.y:
            for parent_a in self.attr:
                # 独依赖属性
                for xi in data[parent_a].unique():
                    for a in self.discrete_attr:
                        if a == parent_a:
                            continue
                        for xj in data[a].unique():
                            _index = np.logical_and(data[a]==xj, data[label]==c, data[parent_a]==xi)
                            self.p_xj_c_xi[c][f'{parent_a}_{xi}'][f'{a}_{xj}'] = (
                                (len(data[_index]) + 1) / (Dc_xi[c][f'{parent_a}_{xi}'] + N_i[a])
                            )
                    for a in self.continuous_attr:
                        if a == parent_a:
                            continue
                        _index = np.logical_and(data[label]==c, data[parent_a]==xi)
                        arg = data[_index][a].mean()
                        sgm = data[_index][a].std()
                        self.p_xj_c_xi[c][f'{parent_a}_{xi}'][a] = OddsDict(u=arg, sigma=sgm)

    def predict(self, sample):
        argmax = None, 0
        for c in self.y:
            print('------------------------------')
            odds = 0
            for parent_a in self.discrete_attr:
                # todo: 连续属性不作为独依赖，待优化
                _o = self.p_c_xi[c][f'{parent_a}_{sample[parent_a]}']
                for a in self.attr:
                    if a == parent_a:
                        continue
                    _xj_odds = (
                        self.p_xj_c_xi[c][f'{parent_a}_{sample[parent_a]}'][f'{a}_{sample[a]}']
                        if a in self.discrete_attr else self.p_xj_c_xi[c][f'{parent_a}_{sample[parent_a]}'][a][sample[a]]
                    )
                    _o *= _xj_odds
                odds += _o
            print(f'odds of label {c}: {odds}')
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

    model = AODEBayes(data, label, discrete_attr, continuous_attr)

    sample = data.iloc[0]
    print('sample 1: \n', sample)
    model.predict(sample)
