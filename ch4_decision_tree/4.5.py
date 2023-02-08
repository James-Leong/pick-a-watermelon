import os
import pandas as pd

from __init__ import *
from ch4_decision_tree.decision_tree import DecisionTreeModel


if __name__ == '__main__':

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    # 数据处理
    label_column = '好瓜'
    labels = ['是', '否']
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

    model = DecisionTreeModel(
        dict(),
        [],
        new_columns,
        label_column,
        labels,
        method='logistic_regression',
    )
    # 无剪枝
    model.generate(data=data)
    print('==========\n无剪枝决策树：')
    model.draw()
    # predict = model.predict(data=validation_data)
    # print(f'预测结果： \n{predict}，验证集准确率：{model.accuracy()}')
