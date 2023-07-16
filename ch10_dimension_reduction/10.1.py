import numpy as np
import pandas as pd

from __init__ import *


class KNN:

    def __init__(self, k, continuous_col, label):
        self.k = k
        self.samples = None
        self.continuous_col = continuous_col
        self.label = label

    def generate(self, data: pd.DataFrame):
        self.samples = data

    def _predict(self, row):
        vector = row[self.continuous_col].values.reshape(1, -1).astype(float)
        samples = self.samples[self.continuous_col].values
        distance = np.linalg.norm(vector-samples, axis=1)
        idx = np.argpartition(distance, self.k)[:self.k]
        result = self.samples.iloc[idx][self.label].value_counts().idxmax()
        return result

    def predict(self, data: pd.DataFrame):
        result = [self._predict(row) for _, row in data.iterrows()]
        return result


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
    data = data.drop(columns=discrete_attr)

    model = KNN(k=5, continuous_col=continuous_attr, label=label)
    model.generate(data=data)
    predict = model.predict(data=data)
    print(predict)
