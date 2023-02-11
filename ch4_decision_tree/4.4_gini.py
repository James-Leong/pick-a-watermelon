import os
import pandas as pd

from __init__ import *
from ch4_decision_tree.decision_tree import DecisionTreeModel


if __name__ == '__main__':

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    # 数据处理
    label_column = '好瓜'
    continuous_columns = ['密度', '含糖率']
    discrete_columns = ['脐部', '色泽', '根蒂', '敲声', '纹理', '触感']
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data = data.drop(columns=continuous_columns)
    data[discrete_columns] = data[discrete_columns].astype(str)
    training_data = data.iloc[[0, 1, 2, 5, 6, 9, 13, 14, 15, 16]]
    validation_data = data.iloc[[3, 4, 7, 8, 10, 11, 12]]

    model = DecisionTreeModel(
        discrete_columns,
        [],
        label_column,
        method='gini',
        validation_data=validation_data,
    )
    # 无剪枝
    model.generate(data=training_data)
    print('==========\n无剪枝决策树：')
    model.draw()
    predict = model.predict(data=validation_data)
    print(f'预测结果： \n{predict}，验证集准确率：{model.accuracy()}')
    # 预剪枝
    model.generate(data=training_data, prune='pre')
    print('==========\n预剪枝决策树：')
    model.draw()
    predict = model.predict(data=validation_data)
    print(f'预测结果： \n{predict}，验证集准确率：{model.accuracy()}')
    # 后剪枝
    model.generate(data=training_data, prune='post')
    print('==========\n后剪枝决策树：')
    model.draw()
    predict = model.predict(data=validation_data)
    print(f'预测结果： \n{predict}，验证集准确率：{model.accuracy()}')

