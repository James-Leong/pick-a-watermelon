
# 与原书不同，原书中一个样本xi为列向量，本代码中一个样本xi为行向量
# 尝试了两种优化方法，梯度下降和牛顿法，两者结果基本相同
# 有时因初始化的原因，会导致牛顿法中海森矩阵为奇异矩阵。处理方式为：当海森矩阵奇异时，重新初始化。
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model


def sigmoid(x):
    """
    sigmoid激活函数

    Args:
        x (np.array): 输入

    Returns:
        np.array: 输出
    """
    s = 1 / (1 + np.exp(-x))
    return s


def initialize_beta(n):
    """
    初始化权重矩阵β, β=(w;b)

    Args:
        n (int): 变量x的维度

    Returns:
        np.array: β矩阵(n+1 × 1)
    """
    # 使用标准正态分布生成权重
    beta = np.random.randn(n + 1, 1) * 0.5 + 1
    return beta


def J_cost(X, y, beta):
    """
    对数几率回归的代价函数，即对数似然函数

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
        beta (np.array): 权重矩阵 (特征维度数+1 × 1)

    Returns:
        np.float64: 对数似然值
    """
    # x_hat = (x;1), 使得 w^T·x + b 可以表示为 β^T·x_hat
    sample = X.shape[0]
    X_hat = np.c_[X, np.ones(shape=(sample, 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # 计算似然
    Lbeta = (-y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))).sum()

    return Lbeta


def gradient(X, y, beta):
    """
    计算似然函数（式3.27）对β的一阶导数（式3.30），从而得到似然函数的梯度

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
        beta (np.array): 权重矩阵 (特征维度数+1 × 1)

    Returns:
        np.array: 梯度 (特征维度数+1, 1)
    """
    # x_hat = (x;1), 使得 w^T·x + b 可以表示为 β^T·x_hat
    x_sample = X.shape[0]
    X_hat = np.c_[X, np.ones(shape=(x_sample, 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # 计算后验概率
    p1 = sigmoid(np.dot(X_hat, beta))
    # sum(0)表示按行相加，得到各个特征的一阶导数值，即梯度
    grad = (-X_hat * (y - p1)).sum(0).reshape(-1, 1)

    return grad


def hessian(X, y, beta):
    """
    计算似然函数（式3.27）对β的二阶导数（式3.31），即海森矩阵

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
        beta (np.array): 权重矩阵 (特征维度数+1 × 1)

    Returns:
        np.array: 海森矩阵 (特征维度数+1 × 特征维度数+1)
    """
    sample = X.shape[0]
    X_hat = np.c_[X, np.ones(shape=(sample, 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))
    P = np.eye(sample) * p1 * (1 - p1)  # 对角矩阵
    assert P.shape[0] == P.shape[1]
    hess = np.dot(np.dot(X_hat.T, P), X_hat)

    return hess


def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    """
    使用梯度下降法优化参数

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
        beta (np.array): 权重矩阵 (特征维度数+1 × 1)
        learning_rate (float): 学习率
        num_iterations (int): 迭代次数
        print_cost (bool): 是否打印代价函数值

    Returns:
        np.array: 权重矩阵
    """
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))

    return beta


def update_parameters_newton(X, y, beta, num_iterations, print_cost):
    """
    使用牛顿法优化参数

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
        beta (np.array): 权重矩阵 (特征维度数+1 × 1)
        num_iterations (int): 迭代次数
        print_cost (bool): 是否打印代价函数值

    Returns:
        np.array: 权重矩阵
    """

    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        hess = hessian(X, y, beta)
        determinant = np.linalg.det(hess)
        if determinant == 0:
            # 重新初始化
            beta = initialize_beta(X.shape[1])
        else:
            beta = beta - np.dot(np.linalg.inv(hess), grad)

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta


def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
    """
    对数几率回归（逻辑回归）模型

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
        num_iterations (int, optional): 迭代次数. Defaults to 100.
        learning_rate (float, optional): 学习率. Defaults to 1.2.
        print_cost (bool, optional): 是否输出代价. Defaults to False.
        method (str, optional): 数值优化算法. Defaults to 'gradDesc'.

    Raises:
        ValueError: 无法识别的数值优化算法

    Returns:
        np.array: 权重矩阵
    """
    sample, feature = X.shape
    beta = initialize_beta(feature)
    # np.random.randn产生的下面这个随机参数会导致牛顿法在第4次迭代时算出的海森矩阵奇异，因此牛顿法需要特殊处理
    # beta = np.array([[0.91187113],
    #    [0.94727392],
    #    [1.8451866 ]]
    # )

    if method == 'gradDesc':
        return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)
    elif method == 'Newton':
        return update_parameters_newton(X, y, beta, num_iterations, print_cost)
    else:
        raise ValueError('Unknown solver %s' % method)


def predict(X, beta):
    """
    预测

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        beta (np.array): 权重矩阵 (特征维度数+1 × 1)

    Returns:
        np.array: 预测结果矩阵
    """
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))

    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0

    return p1


if __name__ == '__main__':

    data_path = './dataset/watermelon3_0_Ch.csv'
    # dataframe转为np.array格式
    data = pd.read_csv(data_path).values

    is_good = data[:, 9] == '是'
    is_bad = data[:, 9] == '否'

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    plt.scatter(data[:, 7][is_good], data[:, 8][is_good], c='k', marker='o')
    plt.scatter(data[:, 7][is_bad], data[:, 8][is_bad], c='r', marker='x')

    plt.xlabel('密度')
    plt.ylabel('含糖量')

    # 可视化 自定义logistic 模型结果
    # optimize_method = 'gradDesc'
    optimize_method = 'Newton'
    beta = logistic_model(X, y, print_cost=True, method=optimize_method, learning_rate=0.3, num_iterations=1000)
    w1, w2, intercept = beta
    # z = w1 * x1 + w2 * x2 + intercept = 0
    x1 = np.linspace(0, 1)
    x2 = -(w1 * x1 + intercept) / w2
    ax1, = plt.plot(x1, x2, label=rf'my_logistic_{optimize_method}')

    # 可视化 sklearn LogisticRegression 模型结果
    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 注意sklearn的逻辑回归中，C越大表示正则化程度越低。
    lr.fit(X, y)
    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(f'sklearn logistic regression cost: {J_cost(X, y, lr_beta)}')
    w1_sk, w2_sk = lr.coef_[0, :]
    x1_sk = np.linspace(0, 1)
    x2_sk = -(w1_sk * x1_sk + lr.intercept_) / w2
    ax2, = plt.plot(x1_sk, x2_sk, label=r'sklearn_logistic')

    plt.legend(loc='upper right')
    plt.show()
