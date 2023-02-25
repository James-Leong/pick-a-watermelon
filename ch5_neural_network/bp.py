import math
import random


class StandardBP:
    def __init__(self) -> None:
        self.input_layer = None
        self.hidden_layer = None  # 隐藏层阈值
        self.output_layer = None  # 输出层阈值
        self.v_weight = None  # 输入层与隐藏层的连接权值
        self.w_weight = None  # 隐藏层与输出层的连接权值
        self.alpha = None  # 隐藏层输入值
        self.b = None  # 隐藏层输出值
        self.beta = None  # 输出层输入值
        self.y = None  # 输出层输出值

    def init(self, input: int, hidden: int, ouput: int):
        self.input_layer = [None for i in range(input)]
        self.hidden_layer = [random.random() for i in range(hidden)]
        self.output_layer = [random.random() for i in range(ouput)]
        self.v_weight = []
        for x in self.input_layer:
            self.v_weight.append([random.random() for b in self.hidden_layer])
        self.w_weight = []
        for b in self.hidden_layer:
            self.w_weight.append([random.random() for y in self.output_layer])

    @staticmethod
    def sigmoid(x):
        s = 1/ (1 + math.exp(-x))
        return s

    @staticmethod
    def mean_square_error(predict, sample):
        """
        求均方误差

        Args:
            predict (_type_): 预测值
            sample (_type_): 样本值

        Returns:
            _type_: 均方误差
        """
        e = (predict - sample) ** 2 / 2
        return e

    def update_w(self, g, eta):
        deta_w = []
        for h in range(len(self.hidden_layer)):
            deta_w.append([
                eta * g[j] * self.b[h]
                for j in range(len(self.output_layer))
            ])
        for h, w_list in enumerate(self.w_weight):
            for j in range(len(w_list)):
                self.w_weight[h][j] += deta_w[h][j]

    def update_thet(self, g, eta):
        deta_thet = [-eta * g[j] for j in range(len(self.output_layer))]
        for j in range(len(self.output_layer)):
            self.output_layer[j] += deta_thet[j]

    def update_v(self, e, x, eta):
        deta_v = []
        for i in range(len(self.input_layer)):
            deta_v.append([
                eta * e[h] * x[i]
                for h in range(len(self.hidden_layer))
            ])
        for i, v_list in enumerate(self.v_weight):
            for h in range(len(v_list)):
                self.v_weight[i][h] += deta_v[i][h]

    def update_lambda(self, e, eta):
        deta_lambda = [-eta * e[h] for h in range(len(self.hidden_layer))]
        for h in range(len(self.hidden_layer)):
            self.hidden_layer[h] += deta_lambda[h]

    def _predict(self, x):
        # 计算隐藏层输入：alpha_h = sigma(v_ih * x_i)
        self.alpha = [0 for h in range(len(self.hidden_layer))]
        for i, v_list in enumerate(self.v_weight):
            for h, v_ih in enumerate(v_list):
                self.alpha[h] += v_ih * x[i]
        # 计算隐藏层输出：b_h = sigmoid(alpha_h - lambda_h)
        self.b = [self.sigmoid(self.alpha[h] - self.hidden_layer[h]) for h in range(len(self.hidden_layer))]
        # 计算输出层输入：beta_j = sigma(w_hj * b_h)
        self.beta = [0 for j in range(len(self.output_layer))]
        for h, w_list in enumerate(self.w_weight):
            for j, w_hj in enumerate(w_list):
                self.beta[j] += w_hj * self.b[h]
        # 计算输出层输出：y_j = sigmoid(beta_j - thet_j)
        self.y = [self.sigmoid(self.beta[j] - self.output_layer[j]) for j in range(len(self.output_layer))]
        return self.y

    def predict(self, X):
        y = []
        for x in X:
            y.append(self._predict(x))
        return y

    def train(self, X, y, eta, n=1000):
        """
        训练神经网络

        Args:
            X (list or np.array): 输入
            y (list or np.array): 输出
            eta (float): 学习率
            n (int): 最大迭代次数
        """
        if len(X) != len(y):
            raise ValueError('参数错误！')
        _cur = 0
        while _cur < n:
            _cur += 1
            for _index, x_sample in enumerate(X):
                y_sample = y[_index]
                y_predict = self._predict(x_sample)[0]
                # error_k = self.mean_square_error(predict=y_predict, sample=y_sample)
                # print(error_k)
                g = [y_predict * (1- y_predict) * (y_sample - y_predict)]
                e = [0 for h in range(len(self.hidden_layer))]
                for h in range(len(e)):
                    e[h] = self.b[h] * (1- self.b[h]) * sum([
                        self.w_weight[h][j] * g[j] for j in range(len(self.output_layer))
                    ])
                self.update_w(g, eta)
                self.update_thet(g, eta)
                self.update_v(e, x_sample, eta)
                self.update_thet(e, eta)
