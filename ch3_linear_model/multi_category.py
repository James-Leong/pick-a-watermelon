import numpy as np
import pandas as pd

from copy import deepcopy

class MultiCategoryModel:

    def __init__(self, method: str, model, ) -> None:
        """
        多分类处理器

        Args:
            method (str): 多分类处理方法，eg: OvO, OvR, MvM
            model (_type_): 模型
        """
        self.model = model
        self.method = method
        self.model_pool = []

    def generate_by_OvR(self, data: pd.DataFrame, *args, **kwargs):
        label_col = self.model.label_col
        self.model._confirm_label_values(data)
        label_values = self.model.labels
        for value in label_values:
            other_value = f'{value}_rest'
            # 训练集处理
            train_data = data.copy()
            train_data.loc[train_data[label_col] != value, label_col] = other_value
            model = deepcopy(self.model)
            # 验证集处理
            if model.validation_data is not None:
                model.validation_data.loc[model.validation_data[label_col] != value, label_col] = other_value
            model.generate(data=train_data, *args, **kwargs)
            # model.draw()
            self.model_pool.append((model, value, other_value))

    def generate(self, data: pd.DataFrame, *args, **kwargs):
        self.model_pool.clear()
        if self.method == 'OvO':
            pass
        elif self.method == 'OvR':
            self.generate_by_OvR(data=data, *args, **kwargs)
        elif self.method == 'MvM':
            pass
        else:
            raise ValueError(f'不支持的多分类处理方法：{self.method}')

    def predict_by_OvR(self, data: pd.DataFrame):
        label_values = self.model.labels
        samples = data.shape[0]
        predict_list = [[] for _ in range(samples)]
        result = []
        # 所有模型都做出预测
        for model, value, other_value in self.model_pool:
            model_predict = model.predict(data)
            for _i, _r in enumerate(model_predict):
                predict_list[_i].append(_r)
        # 根据多个分类器结果确认最终分类结果
        for possible_results in predict_list:
            pos_category = []
            neg_category = []
            for item in possible_results:
                if item.find('_rest') >= 0:
                    neg_category.append(item.replace('_res', ''))
                else:
                    pos_category.append(item)
            if len(pos_category) == 1:
                # 若仅有1个分类器预测为正类，则对应标记即为结果
                result.append(pos_category[0])
            else:
                # 选择置信度最大的类别
                # todo: 预测置信度，暂时用简单判断
                possibility = {value: 0 for value in label_values}
                for value in pos_category:
                    possibility[value] += 1
                for _ in neg_category:
                    for value in [_v for _v in label_values if _v != _]:
                        possibility[value] -= 1
                possibility = sorted(possibility.items(), key=lambda x: x[1], reverse=True)
                result.append([possibility[0][0]])
        return result

    def predict(self, data: pd.DataFrame):
        if self.method == 'OvO':
            pass
        elif self.method == 'OvR':
            return self.predict_by_OvR(data=data)
        elif self.method == 'MvM':
            pass
        else:
            raise ValueError(f'不支持的多分类处理方法：{self.method}')

    def accuracy(self, data: pd.DataFrame):
        predict = self.predict(data=data)
        num = len(data)
        error = 0
        for index, value in enumerate(predict):
            if value != data.iloc[index][self.model.label_col]:
                error += 1
        accuracy = 1 - error / num
        return accuracy
