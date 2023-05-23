import pandas as pd

from __init__ import *
from ch4_decision_tree.decision_tree import DecisionTreeModel
from ch8_ensemble.adaboost import Adaboost


if __name__ == '__main__':

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon3_0_Ch.csv')
    # 数据处理
    label_column = '好瓜'
    continuous_columns = ['密度', '含糖率']
    discrete_columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data = data.drop(columns=discrete_columns)
    data[continuous_columns] = data[continuous_columns].astype(float)

    model = DecisionTreeModel(
        discrete_cols=[],
        continuous_cols=continuous_columns,
        label_col=label_column,
        continuous_reusable=False,  # 如果连续属性可以重复划分，拟合效果过于好了
    )
    ensemble = Adaboost(learner=model, T=1)
    ensemble.generate(data=data)
    predict = ensemble.predict(data=data)
    accuracy = ensemble.accuracy(data=data)
    print(f'预测结果： \n{predict}')
    print(f'准确率：{accuracy}')
