import os
import pandas as pd
from sklearn.utils import shuffle

from __init__ import *
from ch5_neural_network.bp import StandardBP


if __name__ == '__main__':

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    # 数据处理
    label_column = '好瓜'
    continuous_columns = ['密度', '含糖率']
    discrete_columns = ['脐部', '色泽', '根蒂', '敲声', '纹理', '触感']
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data[discrete_columns] = data[discrete_columns].astype(str)
    # one hot encode方式处理离散属性
    data = pd.get_dummies(data, columns=discrete_columns)
    new_columns = [_column for _column in data.columns if _column != label_column]
    data.loc[data[label_column] == '是', label_column] = 1
    data.loc[data[label_column] == '否', label_column] = 0
    data[label_column] = data[label_column].astype(int)
    data = shuffle(data)
    X = data[new_columns].values
    y = data[label_column].values

    standard_bp = StandardBP()
    standard_bp.init(input=len(new_columns), hidden=10, ouput=1)

    standard_bp.train(X=X, y=y, eta=0.01, n=10000)
    _predict = standard_bp.predict(X)
    y_predict = [1 if _p[0] > 0.5 else 0 for _p in _predict]
    error = 0
    for _index, _y in enumerate(y):
        if _y != y_predict[_index]:
            error += 1
    print(f'正确率：{round(1-error/len(y), 2)}')
