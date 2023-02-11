import pandas as pd

from __init__ import *
from ch4_decision_tree.decision_tree import DecisionTreeModel


if __name__ == '__main__':

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    # 数据处理
    label_column = '好瓜'
    continuous_columns = ['密度', '含糖率']
    discrete_columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data[discrete_columns] = data[discrete_columns].astype(str)
    data[continuous_columns] = data[continuous_columns].astype(float)

    model = DecisionTreeModel(discrete_columns, continuous_columns, label_column)
    model.generate(data=data)
    model.draw()
    predict = model.predict(data=data)
    print(f'预测结果： \n{predict}')

