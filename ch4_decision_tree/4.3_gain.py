import pandas as pd

from decision_tree import DecisionTreeModel


if __name__ == '__main__':

    # 读取数据集
    data_path = './dataset/watermelon3_0_Ch.csv'
    # 数据处理
    label_column = '好瓜'
    labels = ['是', '否']
    continuous_columns = ['密度', '含糖率']
    discrete_columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:]
    data[discrete_columns] = data[discrete_columns].astype(str)
    data[continuous_columns] = data[continuous_columns].astype(float)
    # 记录原始测试集的各属性取值范围
    attr_value_map = dict()
    for attr in discrete_columns:
        attr_value_map[attr] = {f'== "{value}"' for value in data[attr].unique()}
    for attr in continuous_columns:
        attr_value_map[attr] = None  # 计算时动态生成

    model = DecisionTreeModel(attr_value_map, discrete_columns, continuous_columns, label_column, labels)
    model.generate(data=data)
    model.draw()
    predict = model.predict(data=data)
    print(f'预测结果： \n{predict}')

