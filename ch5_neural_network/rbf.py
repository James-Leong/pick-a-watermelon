import numpy as np


class RBF:
    """单层径向基网络"""

    def __init__(self) -> None:
        self.w = None  # 隐藏层到输出层的连接权值，（1×神经元个数）
        self.c = None  # 隐藏层的中心，（神经元个数×特征数）
        self.beta = None  # 样本与神经元中心的距离的缩放系数（1×神经元个数）

    def init(self, hidden: int, c: np.array):
        """
        初始化RBF网络

        Args:
            hidden (int): 隐藏层神经元的个数
            c (np.array): 隐藏层的中心，（神经元个数×特征数）
        """
        self.w = np.zeros(shape=(1, hidden))
        self.c = c.reshape(hidden, -1)
        self.beta = np.zeros(shape=(1, hidden))

    def loss(self, y_real, y_predict):
        """
        损失函数

        Args:
            y_real (np.array): 真实值
            y_predict (np.array): 预测值

        Return:
            np.float: 损失值
        """
        return np.sum(np.square(y_predict - y_real)) / 2

    def _predict(self, x):
        """
        前向传播

        Args:
            x (np.array): 样本，（样本数×特征数）

        Returns:
            np.array: 范数，(样本数×神经元个数)
            np.array: 隐藏层输出，（样本数×神经元个数）
            np.array: 输出层输出，（样本数×1）
        """
        samples, features = x.shape
        # 范数，shape=(样本数×神经元个数)
        norm = np.linalg.norm(x.reshape((samples, 1, features)) - self.c, axis=2)
        # 隐藏层输出，使用高斯径向基函数，shape=(样本数×神经元个数)
        rho = np.exp(-self.beta * norm)
        # 输出层输出，是输入的线性组合
        y = np.dot(rho, self.w.T)
        return norm, rho, y

    def predict(self, X):
        """
        预测

        Args:
            x (np.array): 样本，（样本数×特征数）

        Returns:
            np.array: 输出层输出，（样本数×1）
        """
        _, _, y = self._predict(X)
        return y

    def train(self, X, y, eta=0.01, n=1000):
        """
        训练神经网络

        Args:
            X (np.array): 样本集特征
            y (np.array): 样本集标签
            eta (float, optional): 学习率. Defaults to 0.01.
            n (int, optional): 最大迭代次数. Defaults to 1000.
        """
        if len(X) != len(y):
            raise ValueError('参数错误！')
        sample_n = X.shape[0]
        loss_list = []
        for _ in range(n):
            # 前向传播
            norm, rho, y_predict = self._predict(X)
            loss = self.loss(y_real=y, y_predict=y_predict)
            # 误差逆传播
            w_grad = np.dot((y_predict - y).T, rho)
            beta_grad = - np.dot((y_predict - y).T, self.w * rho * norm)
            self.w += -eta * w_grad
            self.beta += -eta * beta_grad
            loss_list.append(loss)
        return loss_list
