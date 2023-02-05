import math
import numpy as np
import pandas as pd

from typing import Tuple


class TreeNode:

    def __init__(self) -> None:
        self.attr = None
        self.label = None
        self.continuous = False
        self.next = dict()

    def predict(self, data: dict):
        if not self.next or not self.attr or self.attr not in data:
            return self.label
        for value, next_node in self.next.items():
            if (self.continuous and eval(f'{data[self.attr]} {value}')) or \
                    (not self.continuous and eval(f'"{data[self.attr]}" {value}')):
                return next_node.predict(data) 
        return self.label        


class DecisionTreeModel:

    def __init__(self, attr_value_map: dict, discrete_cols: list, continuous_cols: list, label_col: str, labels: list) -> None:
        self.attr_value_map = attr_value_map  # 记录各属性的取值范围
        self.discrete_cols = discrete_cols  # 记录离散属性
        self.continuous_cols = continuous_cols  # 记录连续属性
        self.label_col = label_col  # 记录结果标记
        self.labels = labels  # 分类的所有值
        self.root = None  # 根节点

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

    def choose_best_attribute(self, data: pd.DataFrame, available_attrs: list):
        gain_list = []
        for attr in available_attrs:
            gain_a, t = self.Gain_D_a(data, attr)
            gain_list.append((attr, gain_a, t))
        gain_list = sorted(gain_list, key=lambda x: x[1], reverse=True)
        attr_best = gain_list[0][0]
        t_best = gain_list[0][2]
        return attr_best, t_best

    def _generate(self, data: pd.DataFrame, attr: str, available_attrs: list) -> TreeNode:
        # 生成节点
        node = TreeNode()
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
        if attr_best in self.continuous_cols:
            node.continuous = True
            values = (f'<= {t_best}', f'> {t_best}')
        else:
            node.continuous = False
            values = self.attr_value_map[attr_best]
        for value in values:
            # 为属性a的所有可能取值分别生成一个分支，Dv表示这个分支的样本子集
            Dv = data.query(f'{attr_best} {value}')
            if Dv.empty:
                # 已知属性a的一个取值，但样本子集为空，使用当前节点的概率作为子节点的先验分布
                next_node = TreeNode()
                next_node.label = data[self.label_col].value_counts().index[0]
            else:
                next_attrs = available_attrs.copy()
                if not node.continuous:
                    next_attrs.remove(attr_best)
                else:
                    # 与离散属性不同，结点划分为连续属性后，仍可作为其后代节点的划分属性
                    pass
                next_node = self._generate(data=Dv, attr=attr_best, available_attrs=next_attrs)
            node.next[value] = next_node
        return node

    def generate(self, data: pd.DataFrame) -> TreeNode:
        root = self._generate(data=data, attr=None, available_attrs=self.discrete_cols+self.continuous_cols)
        self.root = root
        return root

    def predict(self, data: pd.DataFrame):
        node = self.root
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
        print('|    ' * depth + f'|[{node.attr}]')
        for value, next_node in node.next.items():
            print('|    ' * depth + f'|  {value}')
            self.draw(next_node, depth=depth+1)


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

