import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle

from __init__ import *
from ch5_neural_network.bp import StandardBP


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.close('all')

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    # 数据处理
    label_column = '好瓜'
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data = shuffle(data)
    # 非数属性编码映射
    for title in data.columns:
        if data[title].dtype == 'object':
            encoder = LabelEncoder()
            data[title] = encoder.fit_transform(data[title])
    # 去均值、方差归一化
    ss = StandardScaler()
    X = data.drop(label_column, axis=1)
    sample_n, feature_n = X.shape
    X = ss.fit_transform(X).reshape(sample_n, -1)
    y = data[label_column].values.reshape(sample_n, -1)

    max_iter = 10000
    standard_bp = StandardBP()
    standard_bp.init(input=feature_n, hidden=10, ouput=y.shape[1])

    standard_loss = standard_bp.train(X=X, y=y, eta=0.01, n=max_iter)
    probability = standard_bp.predict(X)
    y_predict = [1 if _p[0] > 0.5 else 0 for _p in probability]
    error = 0
    for _index, _y in enumerate(y.reshape(sample_n)):
        if _y != y_predict[_index]:
            error += 1
    print(f'标准BP网络正确率：{round(1-error/len(y), 2)}')

    ##Loss可视化
    plt.figure()
    plt.plot([i+1 for i in range(max_iter)], standard_loss)
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("standard_bp_loss.png")
