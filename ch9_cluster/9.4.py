import numpy as np
import pandas as pd

from __init__ import *


class KMeanModel:
    """k-mearn cluster"""

    def __init__(self, k, n=100):
        self.k = k
        self.n = n
        self.mean_vector = None

    def init_mean_vector(self, data):
        random_index = np.random.choice(data.shape[0], size=self.k)
        return data[random_index]

    def dist(self, x, u):
        d = np.linalg.norm(x-u)
        return d

    def confirm_cluster(self, x, mean_vector, C):
        # 计算样本xj与各均值向量的距离
        dist_i = [self.dist(x, ui) for ui in mean_vector]
        # 根据距离最近的均值向量确定xj的簇标记
        d_min = dist_i[0]
        d_idx = 0
        for i, dist in enumerate(dist_i):
            if dist <= d_min:
                d_min = dist
                d_idx = i
        # 将样本xj划入相应的簇
        C[d_idx].append(x)

    def generate(self, data):
        # 初始化均值向量
        mean_vector = self.init_mean_vector(data)
        for _ in range(self.n):
            C = [list() for i in range(self.k)]
            for j in range(data.shape[0]):
                # 确定样本xj的簇
                self.confirm_cluster(data[j], mean_vector, C=C)
            # 计算并更新均值向量
            update = False
            for i in range(self.k):
                new_u = np.mean(C[i], axis=0)
                if (new_u != mean_vector[i]).any():
                    mean_vector[i] = new_u
                    update = True
            # 若均值向量均未更新，提前结束迭代
            if not update:
                print(f'第{_}次迭代均值向量未变化，迭代结束.')
                break
        self.mean_vector = mean_vector
        print(mean_vector)

    def predict(self, data):
        C = [list() for i in range(self.k)]
        for j in range(data.shape[0]):
            # 确定样本xj的簇
            self.confirm_cluster(data[j], self.mean_vector, C=C)
        return C


if __name__ == '__main__':

    # 读取数据集
    data_path = os.path.join(os.path.dirname(__file__), '../dataset/watermelon4_0_Ch.csv')
    # 数据处理
    data = pd.read_csv(data_path, header=None).values

    model = KMeanModel(k=3)
    model.generate(data=data)
    predict = model.predict(data=data)
    print(predict)
