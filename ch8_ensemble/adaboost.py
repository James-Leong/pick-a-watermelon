import numpy as np
import pandas as pd

from copy import deepcopy


class Adaboost:
    """Adaboost algorithm"""

    def __init__(self, learner, T, e=0.5) -> None:
        self.base_learner = learner
        self.T = T  # 集成学习的学习器数量
        self.e = e  # 错误率阈值
        # self.learners = [deepcopy(self.base_learner) for i in range(self.T)]
        self.learns = []

    def generate(self, data: pd.DataFrame):
        weight = np.ones(shape=len(data)) / len(data)
        for t in range(self.T):
            model = deepcopy(self.base_learner)
            model.weight_col = 'weight'
            data['weight'] = weight
            model.generate(data)
            G_t = model.predict(data)
            classify = (data[model.label_col] == G_t).values
            # 计算误差率
            e_t = sum(weight[~classify])
            if e_t > self.e:
                break
            # 计算基学习器的系数
            alpha = 0.5 * np.log((1 - e_t) / e_t)
            # 保存基学习器
            self.learns.append((model, alpha))
            # 计算规范化因子
            _yG = np.ones(shape=len(data))  # 分类相同为1
            _yG[~classify] = -1  # 分类不相同为-1
            Z_t = np.dot(weight, np.exp(-alpha * _yG))
            print(f'第{t}个基学习器的误差率为{e_t}，系数为{alpha}')
            # 更新训练集的权值分布
            weight = weight / Z_t * np.exp(-alpha * _yG)

    def predict(self, data: pd.DataFrame):
        p = np.zeros(shape=len(data))
        for model, alpha in self.learns:
            base_predict = model.predict(data)
            G_t = np.array([1 if _p == '是' else -1 for _p in base_predict])
            p += G_t * alpha
        result = ['是' if _p >= 0 else '否' for _p in p]
        return result

    def accuracy(self, data: pd.DataFrame = None):
        if data.empty:
            raise ValueError('请添加数据集')
        predict = self.predict(data=data)
        if len(data) != len(predict):
            raise ValueError()
        num = len(data)
        error = 0
        for index, value in enumerate(predict):
            if value != data.iloc[index][self.base_learner.label_col]:
                error += 1
        accuracy = 1 - error / num
        return accuracy
