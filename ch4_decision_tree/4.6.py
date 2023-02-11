import numpy as np
import pandas as pd
import warnings

from sklearn import datasets
from sklearn.model_selection import train_test_split

from __init__ import *
from ch3_linear_model.multi_category import MultiCategoryModel
from ch4_decision_tree.decision_tree import DecisionTreeModel


if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    iris = datasets.load_iris()
    discrete_columns = []
    continuous_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label_column = 'category'
    data = pd.DataFrame(iris['data'], columns=continuous_columns)
    data[label_column] = pd.Series(iris['target_names'][iris['target']])

    # 取30个样本为测试集
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=15)
    # 剩下120个样本中，取30个作为剪枝时的验证集
    data_train, data_val = train_test_split(data, test_size=0.25, random_state=15)

    # # 增益率决策树
    # model = DecisionTreeModel(
    #     discrete_cols=discrete_columns,
    #     continuous_cols=continuous_columns,
    #     label_col=label_column,
    #     method='gain',
    #     validation_data=data_val,
    # )
    # # 无剪枝
    # model.generate(data=data_train, prune=None)
    # accuracy = model.accuracy(data=data_test)
    # print('==========\n无剪枝决策树：')
    # model.draw()
    # print(f'gain无剪枝测试集准确率：{accuracy}')
    # # 预剪枝
    # model.generate(data=data_train, prune='pre')
    # accuracy = model.accuracy(data=data_test)
    # print('==========\n预剪枝决策树：')
    # model.draw()
    # print(f'gain预剪枝测试集准确率：{accuracy}')
    # # 后剪枝
    # model.generate(data=data_train, prune='post')
    # accuracy = model.accuracy(data=data_test)
    # print('==========\n后剪枝决策树：')
    # model.draw()
    # print(f'gain后剪枝测试集准确率：{accuracy}')

    # # 基尼指数决策树
    # model = DecisionTreeModel(
    #     discrete_cols=discrete_columns,
    #     continuous_cols=continuous_columns,
    #     label_col=label_column,
    #     method='gini',
    #     validation_data=data_val,
    # )
    # # 无剪枝
    # model.generate(data=data_train, prune=None)
    # accuracy = model.accuracy(data=data_test)
    # print('==========\n无剪枝决策树：')
    # model.draw()
    # print(f'gini无剪枝测试集准确率：{accuracy}')
    # # 预剪枝
    # model.generate(data=data_train, prune='pre')
    # accuracy = model.accuracy(data=data_test)
    # print('==========\n预剪枝决策树：')
    # model.draw()
    # print(f'gini预剪枝测试集准确率：{accuracy}')
    # # 后剪枝
    # model.generate(data=data_train, prune='post')
    # accuracy = model.accuracy(data=data_test)
    # print('==========\n后剪枝决策树：')
    # model.draw()
    # print(f'gini后剪枝测试集准确率：{accuracy}')

    # 对率回归决策树
    model = DecisionTreeModel(
        discrete_cols=discrete_columns,
        continuous_cols=continuous_columns,
        label_col=label_column,
        method='logistic_regression',
        validation_data=data_val,
    )
    multi_category_model = MultiCategoryModel(
        model=model,
        method='OvR'
    )
    multi_category_model.generate(data=data_train)
    accuracy = multi_category_model.accuracy(data=data_test)
    print(f'logistic_regression无剪枝测试集准确率：{accuracy}')

    multi_category_model.generate(data=data_train, prune='pre')
    accuracy = multi_category_model.accuracy(data=data_test)
    print(f'logistic_regression预剪枝测试集准确率：{accuracy}')

    multi_category_model.generate(data=data_train, prune='post')
    accuracy = multi_category_model.accuracy(data=data_test)
    print(f'logistic_regression后剪枝测试集准确率：{accuracy}')
