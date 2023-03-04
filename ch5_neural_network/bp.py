import numpy as np


class StandardBP:
    def __init__(self) -> None:
        self.input_layer = None
        self.hidden_layer = None  # 隐藏层阈值
        self.output_layer = None  # 输出层阈值
        self.v_weight = None  # 输入层与隐藏层的连接权值
        self.w_weight = None  # 隐藏层与输出层的连接权值

    def init(self, input: int, hidden: int, ouput: int):
        self.input_layer = [None for i in range(input)]
        self.hidden_layer = np.zeros((1, hidden))  # 隐藏层阈值，列向量
        self.output_layer = np.zeros((1, ouput))  # 输出层阈值，列向量
        self.v_weight = np.zeros((input, hidden))  # 输入层与隐藏层的连接权值
        self.w_weight = np.zeros((hidden, ouput))  # 隐藏层与输出层的连接权值

    @staticmethod
    def sigmoid(x):
        s = 1/ (1 + np.exp(-x))
        return s

    @staticmethod
    def mean_square_error(predict, sample):
        """
        求均方误差

        Args:
            predict (np.array): 预测值
            sample (np.array): 样本值

        Returns:
            np.array: 均方误差
        """
        # E_k = sigma_j(predict_j - y_j) / 2
        e = np.sum(np.square(predict - sample) / 2, axis=1)
        return e

    def update_w(self, b, g, eta):
        self.w_weight += (eta * g * b).T

    def update_thet(self, g, eta):
        self.output_layer += -eta * g.T

    def update_v(self, e, x, eta):
        self.v_weight += eta * e * x.T

    def update_lambda(self, e, eta):
        self.hidden_layer += -eta * e

    def _predict(self, x):
        """
        前向传播

        Args:
            x (np.array): 样本集，shape=(samples, features)

        Returns:
            np.array: 隐藏层输入
            np.array: 隐藏层输出
            np.array: 输出层输入
            np.array: 输出层输出
        """
        # 计算隐藏层输入：alpha_h = sigma(v_ih * x_i)
        alpha = np.dot(x, self.v_weight)
        # 计算隐藏层输出：b_h = sigmoid(alpha_h - lambda_h)
        b = self.sigmoid(alpha - self.hidden_layer)
        # 计算输出层输入：beta_j = sigma(w_hj * b_h)
        beta = np.dot(b, self.w_weight)
        # 计算输出层输出：y_j = sigmoid(beta_j - thet_j)
        y = self.sigmoid(beta - self.output_layer)
        return alpha, b, beta, y

    def predict(self, X):
        _, _ , _, y = self._predict(X)
        return y

    def train(self, X, y, eta, n=1000):
        """
        训练神经网络

        Args:
            X (list or np.array): 输入
            y (list or np.array): 输出
            eta (float): 学习率
            n (int): 最大迭代次数

        Returns:
            list: 累积误差
        """
        if len(X) != len(y):
            raise ValueError('参数错误！')
        sample_n = X.shape[0]
        error_list = []
        for _ in range(n):
            error = 0  # 累积误差
            for _index in range(sample_n):
                x_sample = X[_index].reshape(1, -1)  # 行向量
                y_sample = y[_index].reshape(1, -1)  # 行向量
                # 前向传播
                _, b, _, y_predict = self._predict(x_sample)
                error_k = self.mean_square_error(predict=y_predict, sample=y_sample)
                error += error_k[0]
                # 误差逆传播
                g = y_predict * (1- y_predict) * (y_sample - y_predict)
                e = b * (1 - b) * np.dot(self.w_weight, g.T).T
                self.update_w(b, g, eta)
                self.update_thet(g, eta)
                self.update_v(e, x_sample, eta)
                self.update_lambda(e, eta)
            error_list.append(error)
        return error_list


class AccumBP(StandardBP):

    def update_w(self, b, g, eta):
        self.w_weight += eta * np.dot(b.T, g)

    def update_thet(self, g, eta):
        self.output_layer += -eta * np.sum(g, axis=0).reshape(1, -1)

    def update_v(self, e, x, eta):
        self.v_weight += eta * np.dot(x.T, e)

    def update_lambda(self, e, eta):
        self.hidden_layer += -eta * np.sum(e, axis=0).reshape(1, -1)

    def train(self, X, y, eta, n=1000):
        """
        训练神经网络

        Args:
            X (list or np.array): 输入
            y (list or np.array): 输出
            eta (float): 学习率
            n (int): 最大迭代次数

        Returns:
            list: 累积误差
        """
        if len(X) != len(y):
            raise ValueError('参数错误！')
        error_list = []
        for _ in range(n):
            # 前向传播
            _, b, _, y_predict = self._predict(X)
            error_k = self.mean_square_error(predict=y_predict, sample=y)
            # 误差逆传播
            g = y_predict * (1- y_predict) * (y - y_predict)
            e = b * (1 - b) * np.dot(self.w_weight, g.T).T
            self.update_w(b, g, eta)
            self.update_thet(g, eta)
            self.update_v(e, X, eta)
            self.update_lambda(e, eta)
            error_list.append(np.sum(error_k))
        return error_list


class AdaptedBP(StandardBP):
    """使用自适应学习率的标准BP算法"""

    def train(self, X, y, eta, n=1000):
        """
        训练神经网络

        Args:
            X (list or np.array): 输入
            y (list or np.array): 输出
            eta (float): 学习率
            n (int): 最大迭代次数

        Returns:
            list: 累积误差
        """
        if len(X) != len(y):
            raise ValueError('参数错误！')
        sample_n = X.shape[0]
        error_list = []
        for _ in range(n):
            error = 0  # 累积误差
            for _index in range(sample_n):
                x_sample = X[_index].reshape(1, -1)  # 行向量
                y_sample = y[_index].reshape(1, -1)  # 行向量
                # 前向传播
                _, b, _, y_predict = self._predict(x_sample)
                error_k = self.mean_square_error(predict=y_predict, sample=y_sample)
                error += error_k[0]
                # 误差逆传播
                g = y_predict * (1- y_predict) * (y_sample - y_predict)
                e = b * (1 - b) * np.dot(self.w_weight, g.T).T
                self.update_w(b, g, eta)
                self.update_thet(g, eta)
                self.update_v(e, x_sample, eta)
                self.update_lambda(e, eta)
            if error_list:
                if error < 0.9999 * error_list[-1]:
                    # 误差显著降低了，可以适当增大学习率
                    eta *= 1.05
                elif error > 1.0001 * error_list[-1]:
                    # 误差没有显著降低，可以适当减小学习率
                    eta *= 0.7
                else:
                    eta *= 1
            error_list.append(error)
        return error_list
