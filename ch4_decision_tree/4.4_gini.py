import os
import pandas as pd

from decision_tree import DecisionTreeModel


if __name__ == '__main__':

    # 读取数据集
    data_path = f'{os.path.dirname(__file__)}/../dataset/watermelon3_0_Ch.csv'
    # 数据处理
    label_column = '好瓜'
    labels = ['是', '否']
    continuous_columns = ['密度', '含糖率']
    discrete_columns = ['脐部', '色泽', '根蒂', '敲声', '纹理', '触感']
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data = data.drop(columns=continuous_columns)
    data[discrete_columns] = data[discrete_columns].astype(str)
    training_data = data.iloc[[0, 1, 2, 5, 6, 9, 13, 14, 15, 16]]
    validation_data = data.iloc[[3, 4, 7, 8, 10, 11, 12]]
    # 记录原始测试集的各属性取值范围
    attr_value_map = dict()
    for attr in discrete_columns:
        attr_value_map[attr] = {f'== "{value}"' for value in data[attr].unique()}

    model = DecisionTreeModel(attr_value_map, discrete_columns, [], label_column, labels, method='gini')
    model.generate(data=training_data)
    print('决策树：')
    model.draw()
    predict = model.predict(data=validation_data)
    print(f'预测结果： \n{predict}')

