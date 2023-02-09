import math
import numpy as np
import pandas as pd

from typing import Tuple

from ch3_linear_model.logistic_regression import logistic_model, predict


class TreeNode:

    def __init__(self) -> None:
        self.attr = None
        self.label = None
        self.continuous = False
        self.next = dict()

    @property
    def leaf(self):
        if self.next:
            return False
        else:
            return True

    def predict(self, data: dict):
        if not self.next or not self.attr or self.attr not in data:
            return self.label
        for value, next_node in self.next.items():
            if (self.continuous and eval(f'{data[self.attr]} {value}')) or \
                    (not self.continuous and eval(f'"{data[self.attr]}" {value}')):
                return next_node.predict(data) 
        return self.label        


class DecisionTreeModel:

    def __init__(
            self, attr_value_map: dict,
            discrete_cols: list,
            continuous_cols: list,
            label_col: str,
            labels: list,
            method: str = 'gain',
            validation_data: pd.DataFrame = None,
    ) -> None:
        """
        决策树模型

        Args:
            attr_value_map (dict): 各属性取值范围
            discrete_cols (list): 离散属性
            continuous_cols (list): 连续属性
            label_col (str): 分类标签
            labels (list): 分类值
            method (str, optional): 属性划分方法. Defaults to 'gain'.
            validation_data (pd.DataFrame, optional): 验证集
        """
        self.attr_value_map = attr_value_map  # 记录各属性的取值范围
        self.discrete_cols = discrete_cols  # 记录离散属性
        self.continuous_cols = continuous_cols  # 记录连续属性
        self.attr_cols = discrete_cols or [] + continuous_cols or []
        self.label_col = label_col  # 记录结果标记
        self.labels = labels  # 分类的所有值
        self.root = None  # 根节点
        self.method = method
        self.validation_data = validation_data

    def Ent(self, data: pd.DataFrame) -> float:
        ent = 0
        num = len(data)
        for k in self.labels:
            pk = len(data.loc[data[self.label_col] == k]) / num
            if pk != 0:
                ent -= pk * math.log(pk, 2)
        return ent

    def Gain_D_a_t(self, data, attr, t) -> float:
        ent_D = self.Ent(data=data)
        D_num = len(data)
        Dv_gt = data.loc[data[attr] > t]
        Dv_num_gt = len(Dv_gt)
        ent_gt = self.Ent(Dv_gt) if Dv_num_gt else 0
        Dv_lte = data.loc[data[attr] <= t]
        Dv_num_lte = len(Dv_lte)
        ent_lte = self.Ent(Dv_lte) if Dv_num_lte else 0

        result = ent_D - (Dv_num_lte / D_num * ent_lte) - (Dv_num_gt / D_num * ent_gt)
        return result

    def _Gain_D_a(self, data, attr) -> float:
        ent_D = self.Ent(data=data)
        D_num = len(data)
        result = ent_D
        for value in self.attr_value_map[attr]:
            Dv = data.query(f'{attr} {value}')
            Dv_num = len(Dv)
            ent_Dv = self.Ent(data=Dv) if Dv_num else 0

            result -= Dv_num / D_num * ent_Dv
        return result

    def Gain_D_a(self, data, attr) -> Tuple:
        if attr in self.discrete_cols:
            return self._Gain_D_a(data, attr=attr), None
        # 计算属性为连续值的增益值
        gain_list = []
        a_values = np.sort(data[attr].unique())
        Ta = ((a_values + np.roll(a_values, -1)) / 2)[:-1]
        for t in Ta:
            gain_list.append((t, self.Gain_D_a_t(data, attr, t)))
        gain_list = sorted(gain_list, key=lambda x: x[1], reverse=True)
        t_best = gain_list[0][0]
        result = gain_list[0][1]
        return  result, t_best

    def Gini_D(self, data: pd.DataFrame) -> float:
        result = 1
        num = len(data)
        for k in self.labels:
            pk = len(data.loc[data[self.label_col] == k]) / num
            result -= pk ** 2
        return result

    def Gini_D_a(self, data: pd.DataFrame, attr: str):
        D_num = len(data)
        result = 0
        for value in self.attr_value_map[attr]:
            Dv = data.query(f'{attr} {value}')
            Dv_num = len(Dv)

            result += Dv_num / D_num * self.Gini_D(data=Dv) if Dv_num else 0
        return result, None

    def split_data(self, data: pd.DataFrame, attr, value):
        if self.method == 'logistic_regression':
            _X = data[self.attr_cols].values
            operator, boundary = value.split(' ')
            boundary = float(boundary)
            _y = predict(X=_X, beta=attr, y=boundary)
            if operator == '<=':
                new_data = data[_y <= boundary]
            elif operator == '>':
                new_data = data[_y > boundary]
            else:
                raise ValueError(f'不支持的操作符：{operator}')
        else:
            new_data = data.query(f'{attr_best} {value}')
        return new_data

    def choose_best_attribute(self, data: pd.DataFrame, available_attrs: list):
        params = []
        if self.method == 'gain':
            for attr in available_attrs:
                gain_a, t = self.Gain_D_a(data, attr)
                params.append((attr, gain_a, t))
            # 选择使得信息增益最大的属性
            params = sorted(params, key=lambda x: x[1], reverse=True)
        elif self.method == 'gini':
            for attr in available_attrs:
                gini_a, _ = self.Gini_D_a(data, attr)
                params.append((attr, gini_a, _))
            # 选择使得Gini指数最小的属性
            params = sorted(params, key=lambda x: x[1], reverse=False)
        elif self.method == 'logistic_regression':
            X = data[self.attr_cols].values
            y = data[self.label_col].values
            beta = logistic_model(X, y, print_cost=False, method='gradDesc', learning_rate=0.3, num_iterations=1000)
            params = [(beta, 0, 0.5)]
        else:
            raise ValueError(f'不支持的划分方法：{self.method}')
        attr_best = params[0][0]
        t_best = params[0][2]
        return attr_best, t_best

    def _generate(
            self,
            data: pd.DataFrame,
            attr: str = None,
            available_attrs: list = None,
            prune: str = None,
            node: TreeNode = None,
    ) -> TreeNode:
        # 生成节点
        node = node or TreeNode()
        node.label = data[self.label_col].value_counts().index[0]
        # 如果样本同属一个类别，返回
        if data[self.label_col].nunique() == 1:
            node.label = data[self.label_col].iloc[0]
            return node
        # 如果属性集为空或者样本在属性上的取值相同，返回
        if not available_attrs or len(available_attrs) <= 1:
            node.label = data[self.label_col].value_counts().index[0]
            return node
        # 选择最优划分属性
        attr_best, t_best = self.choose_best_attribute(data, available_attrs)
        node.attr = attr_best
        if self.method == 'logistic_regression' or attr_best in self.continuous_cols:
            node.continuous = True
            values = (f'<= {t_best}', f'> {t_best}')
        else:
            node.continuous = False
            values = self.attr_value_map[attr_best]
        # 获取当前节点的特征值和样本子集
        features = [(value, self.split_data(data, attr=attr_best, value=value)) for value in values]
        # 划分前的验证集精度
        pre_accuracy = self.accuracy() if prune is not None else 0
        # 对当前节点执行分枝
        for value, Dv in features:
            # 为属性a的所有可能取值分别生成一个分支，Dv表示这个分支的样本子集
            next_node = TreeNode()
            if Dv.empty: 
                # 已知属性a的一个取值，但样本子集为空，使用当前节点样本集中最多的分类作为子节点的先验分布
                next_node.label = data[self.label_col].value_counts().index[0]
            else:
                # 使用样本子集中最多的分类作为子节点的标记
                next_node.label = Dv[self.label_col].value_counts().index[0]
            node.next[value] = next_node
        # 划分后的验证集精度
        cur_accuracy = self.accuracy() if prune in ['pre', 'post'] else 0
        # 预剪枝
        if prune == 'pre' and cur_accuracy <= pre_accuracy:
            # 精度无提升，剪枝
            node.next = dict()
            return node
        # 处理下一代子节点
        for value, Dv in features: 
            if Dv.empty:
                # 已知属性a的一个取值，但样本子集为空，使用当前节点的概率作为子节点的先验分布
                continue
            if self.method == 'logistic_regression':
                # 使用对数回归的多变量决策树，使用全部属性
                next_attrs = available_attrs
            elif node.continuous:
                # 与离散属性不同，结点划分为连续属性后，仍可作为其后代节点的划分属性
                next_attrs = available_attrs
            else:
                next_attrs = available_attrs.copy()
                next_attrs.remove(attr_best)
            self._generate(
                data=Dv,
                attr=attr_best,
                available_attrs=next_attrs,
                prune=prune,
                node=node.next[value],
            )
            # 生成子树后的验证集精度
            post_accuracy = self.accuracy() if prune == 'post' else 0
            # 后剪枝
            if prune == 'post' and post_accuracy < cur_accuracy:
                # 根据奥卡姆剃刀准则应该是<=，这里为了与原书保持一致使用<
                node.next[value].next = dict()
        return node

    def generate(self, data: pd.DataFrame, prune: str = None) -> TreeNode:
        """
        生成决策树

        Args:
            data (pd.DataFrame): 训练集
            prune (str, optional): 是否剪枝. 默认为None. 'pre': 预剪枝. 'post': 后剪枝

        Returns:
            TreeNode: _description_
        """
        root = TreeNode()
        self.root = root
        self._generate(
            data=data, available_attrs=self.attr_cols, prune=prune, node=root,
        )
        return root

    def accuracy(self):
        if self.validation_data is None:
            raise ValueError('请添加验证集')
        predict = self.predict(data=self.validation_data, node=self.root)
        if len(self.validation_data) != len(predict):
            raise ValueError()
        num = len(self.validation_data)
        error = 0
        for index, value in enumerate(predict):
            if value != self.validation_data.iloc[index][self.label_col]:
                error += 1
        accuracy = 1 - error / num
        return accuracy

    def predict(self, data: pd.DataFrame, node: TreeNode = None):
        node = node or self.root
        result = []
        for i in range(len(data)):
            _result = node.predict(data.iloc[i].to_dict())
            result.append(_result)
        return result

    def draw(self, node: TreeNode = None, depth: int = 0):
        if not node:
            node = self.root
        if not node.next:
            print('|    ' * depth + f'| -> {node.label}')
            return
        if self.method == 'logistic_regression':
            _attr = ' + '.join([f'{w}*{self.attr_cols[i]}' for i, w in enumerate(node.attr.flatten()[:-1])])
            _attr += f' + {node.attr.flatten()[-1]}'
            print('|    ' * depth + f'|[sigmoid( {_attr} )]')
        else:
            print('|    ' * depth + f'|[{node.attr}]')
        for value, next_node in node.next.items():
            print('|    ' * depth + f'|  {value}')
            self.draw(next_node, depth=depth+1)
