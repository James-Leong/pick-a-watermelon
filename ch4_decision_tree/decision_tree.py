import math
import numpy as np
import pandas as pd

from typing import Tuple

from ch3_linear_model.logistic_regression import logistic_model, predict, sigmoid


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

    def predict_regression(self, data: np.array):
        predict = np.zeros(len(data))
        if not self.next or self.attr is None:
            predict[:] = self.label
            return predict
        beta = self.attr  # shape=(features+1, 1)
        sample = data.shape[0]
        y = sigmoid(np.c_[data, np.ones(shape=(sample, 1))] @ beta).reshape((sample,))
        for value, next_node in self.next.items():
            operator, t = value.split(' ')
            if operator == '<=':
                _index = y <= float(t)
            else:
                _index = y > float(t)
            next_predict = next_node.predict_regression(data[_index])
            predict[_index] = next_predict
        return predict


class DecisionTreeModel:

    def __init__(
            self,
            discrete_cols: list,
            continuous_cols: list,
            label_col: str, 
            method: str = 'gain',
            attr_value_map: dict = None,
            attr_cols: list = None,
            labels: list = None,
            validation_data: pd.DataFrame = None,
            continuous_reusable: bool = True,
            weight_col: str = None,
    ) -> None:
        """
        决策树模型

        Args:
            
            discrete_cols (list): 离散属性
            continuous_cols (list): 连续属性
            label_col (str): 分类标签
            method (str, optional): 属性划分方法. Defaults to 'gain'.
            attr_value_map (dict): 离散属性取值范围，如果未传，则为数据集中的取值范围
            attr_cols (list): 所有属性列
            labels (list): 分类值取值范围
            validation_data (pd.DataFrame, optional): 验证集
            continuous_reusable (bool, optional): 连续属性是否可服用. Defaults to True.
            weight_col (str, optional): 样本权值标签. Defaults to None.
        """
        self.attr_value_map = attr_value_map  # 记录各属性的取值范围
        self.discrete_cols = discrete_cols or []  # 记录离散属性
        self.continuous_cols = continuous_cols or [] # 记录连续属性
        self.attr_cols = attr_cols or (self.discrete_cols + self.continuous_cols)
        self.label_col = label_col  # 记录结果标记
        self.labels = labels  # 分类的所有值
        self.root = None  # 根节点
        self.method = method
        self.label_map = dict()
        self.reverse_label_map = dict()
        self.validation_data = validation_data
        self.continuous_reusable = continuous_reusable
        self.weight_col = weight_col

    def _confirm_attr_values(self, data: pd.DataFrame):
        """
        记录原始测试集的各属性取值范围

        Args:
            data (pd.DataFrame): 数据集
        """
        if not self.attr_value_map:
            self.attr_value_map = dict()
        for attr in self.discrete_cols:
            if attr not in self.attr_value_map:
                self.attr_value_map[attr] = set()
            for value in data[attr].unique():
                self.attr_value_map[attr].add(f'== "{value}"')

    def _confirm_label_values(self, data: pd.DataFrame):
        if not self.labels:
            self.labels = set()
        for value in data[self.label_col].unique():
            self.labels.add(value)

    def _transform_label(self, data: pd.DataFrame):
        if self.method != 'logistic_regression' or data is None:
            return data
        labels = data[self.label_col].unique()
        if not len(labels):
            return data
        for value in labels:
            if not (isinstance(value, str) or isinstance(value, np.str_)):
                continue
            if not self.label_map:
                self.label_map[value] = 0
            elif value not in self.label_map:
                self.label_map[value] = max(self.label_map.values()) + 1
            else:
                pass
        if not self.label_map:
            return data
        data = data.copy()
        for value, _i in self.label_map.items():
            data.loc[data[self.label_col] == value, self.label_col] = _i
        data[self.label_col] = data[self.label_col].astype(int)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        return data

    def Ent(self, data: pd.DataFrame) -> float:
        """计算当前样本集合的信息熵"""
        ent = 0
        num = len(data) if not self.weight_col else sum(data[self.weight_col].values)
        for k in self.labels:
            k_sample = data.loc[data[self.label_col] == k]
            specific = len(k_sample) if not self.weight_col else sum(k_sample[self.weight_col].values)
            pk = specific / num
            if pk != 0:
                ent -= pk * math.log(pk, 2)
        return ent

    def _Gain_D_a_t(self, data, attr, t) -> float:
        """计算样本集合在连续属性上基于划分点二分后的信息增益"""
        ent_D = self.Ent(data=data)
        D_num = len(data) if not self.weight_col else sum(data[self.weight_col].values)
        Dv_gt = data.loc[data[attr] > t]
        Dv_num_gt = len(Dv_gt) if not self.weight_col else sum(Dv_gt[self.weight_col].values)
        ent_gt = self.Ent(Dv_gt) if Dv_num_gt else 0
        Dv_lte = data.loc[data[attr] <= t]
        Dv_num_lte = len(Dv_lte) if not self.weight_col else sum(Dv_lte[self.weight_col].values)
        ent_lte = self.Ent(Dv_lte) if Dv_num_lte else 0

        result = ent_D - (Dv_num_lte / D_num * ent_lte) - (Dv_num_gt / D_num * ent_gt)
        return result

    def _Gain_D_a_cont(self, data, attr) -> Tuple:
        """计算样本集合在连续属性上的信息增益，选择最优的划分点对样本集合进行划分"""
        gain_list = []
        a_values = np.sort(data[attr].unique())
        if len(a_values) == 1:
            return self._Gain_D_a_t(data, attr=attr, t=a_values[0]), a_values[0]
        Ta = ((a_values + np.roll(a_values, -1)) / 2)[:-1]
        for t in Ta:
            gain_list.append((t, self._Gain_D_a_t(data, attr, t)))
        gain_list = sorted(gain_list, key=lambda x: x[1], reverse=True)
        t_best = gain_list[0][0]
        result = gain_list[0][1]
        return result, t_best

    def _Gain_D_a_disc(self, data, attr) -> float:
        """计算样本集合在离散属性上的信息增益"""
        ent_D = self.Ent(data=data)  # 样本集合总的信息熵
        D_num = len(data) if not self.weight_col else sum(data[self.weight_col].values)
        result = ent_D
        for value in self.attr_value_map[attr]:
            Dv = data.query(f'{attr} {value}')
            Dv_num = len(Dv) if not self.weight_col else sum(Dv[self.weight_col].values)
            ent_Dv = self.Ent(data=Dv) if Dv_num else 0  # 样本集合在属性a上取值为value的子集的信息熵

            result -= Dv_num / D_num * ent_Dv
        return result

    def Gain_D_a(self, data: pd.DataFrame, attr: str) -> Tuple:
        """
        计算样本集合在属性attr上的信息增益

        Args:
            data (pd.DataFrame): 样本集
            attr (str): 属性名称

        Returns:
            float: 信息增益值
            float: 连续属性的最佳划分值
        """
        if attr in self.discrete_cols:
            return self._Gain_D_a_disc(data, attr=attr), None
        else:
            return self._Gain_D_a_cont(data, attr=attr)

    def Gini_D(self, data: pd.DataFrame) -> float:
        result = 1
        num = len(data)
        for k in self.labels:
            pk = len(data.loc[data[self.label_col] == k]) / num
            result -= pk ** 2
        return result

    def Gini_D_a_t(self, data: pd.DataFrame, attr: str, t: float) -> float:
        D_num = len(data)
        # 属性值小于等于t
        Dv_lte = data.loc[data[attr] <= t]
        Dv_num_lte = len(Dv_lte)
        gini_lte = self.Gini_D(Dv_lte) if Dv_num_lte else 0
        # 属性值大于t
        Dv_gt = data.loc[data[attr] > t]
        Dv_num_gt = len(Dv_gt)
        gini_gte = self.Gini_D(Dv_gt) if Dv_num_gt else 0

        result = Dv_num_lte / D_num * gini_lte + Dv_num_gt / D_num * gini_gte
        return result

    def _Gini_D_a(self, data: pd.DataFrame, attr: str):
        D_num = len(data)
        result = 0
        for value in self.attr_value_map[attr]:
            Dv = data.query(f'{attr} {value}')
            Dv_num = len(Dv)

            result += Dv_num / D_num * self.Gini_D(data=Dv) if Dv_num else 0
        return result, None

    def Gini_D_a(self, data: pd.DataFrame, attr: str):
        if attr in self.discrete_cols:
            return self._Gini_D_a(data, attr=attr), None
        # 计算属性为连续值的增益值
        gini_list = []
        a_values = np.sort(data[attr].unique())
        if len(a_values) == 1:
            return self.Gini_D_a_t(data, attr=attr, t=a_values[0]), a_values[0]
        Ta = ((a_values + np.roll(a_values, -1)) / 2)[:-1]
        for t in Ta:
            gini_list.append((t, self.Gini_D_a_t(data, attr, t)))
        gini_list = sorted(gini_list, key=lambda x: x[1])
        # 属性a的gini值，所有t可能取值中，使得gini值最小的t划分的值
        t_best = gini_list[0][0]
        result = gini_list[0][1]
        return  result, t_best

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
            new_data = data.query(f'{attr} {value}')
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
                if len(Dv) == len(data):
                    # 回归模型把所有样本都划分为同一类
                    continue
                # 使用对数回归的多变量决策树，使用全部属性
                next_attrs = available_attrs
            elif node.continuous and self.continuous_reusable:
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
        self._confirm_attr_values(data=data)
        self._confirm_label_values(data=data)
        data = self._transform_label(data=data)
        self.validation_data = self._transform_label(self.validation_data)
        root = TreeNode()
        self.root = root
        self._generate(
            data=data, available_attrs=self.attr_cols, prune=prune, node=root,
        )
        return root

    def accuracy(self, data: pd.DataFrame = None):
        if data is None and self.validation_data is not None:
            data = self.validation_data
        elif data is None and self.validation_data is None:
            raise ValueError('请添加验证集')
        if data.empty:
            raise ValueError('请添加数据集')
        predict = self.predict(data=data, node=self.root)
        if len(data) != len(predict):
            raise ValueError()
        num = len(data)
        error = 0
        for index, value in enumerate(predict):
            if value != data.iloc[index][self.label_col]:
                error += 1
        accuracy = 1 - error / num
        return accuracy

    def predict(self, data: pd.DataFrame, node: TreeNode = None):
        node = node or self.root
        if self.method == 'logistic_regression':
            result = node.predict_regression(data[self.attr_cols].values).tolist()
        else:
            result = []
            for i in range(len(data)):
                _result = node.predict(data.iloc[i].to_dict())
                result.append(_result)
        if self.reverse_label_map:
            result = [self.reverse_label_map.get(_r) for _r in result]
        return result

    def draw(self, node: TreeNode = None, depth: int = 0):
        if not node:
            node = self.root
        if not node.next:
            # 输出结果
            if self.reverse_label_map and node.label in self.reverse_label_map:
                label = self.reverse_label_map[node.label]
            else:
                label = node.label
            print('|    ' * depth + f'| -> {label}')
            return
        # 判断条件
        if self.method == 'logistic_regression':
            _attr = ' + '.join([f'{w}*{self.attr_cols[i]}' for i, w in enumerate(node.attr.flatten()[:-1])])
            _attr += f' + {node.attr.flatten()[-1]}'
            print('|    ' * depth + f'|[sigmoid( {_attr} )]')
        else:
            print('|    ' * depth + f'|[{node.attr}]')
        # 值
        for value, next_node in node.next.items():
            print('|    ' * depth + f'|  {value}')
            # 绘制子节点
            self.draw(next_node, depth=depth+1)
