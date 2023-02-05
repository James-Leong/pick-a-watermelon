# 编程实现线性判别分析，并给出西瓜数据集3.0a 上的结果
import numpy as np
import pandas as pd


def LDA_model(X, y):
    """
    线性判别分析

    Args:
        X (np.array): 样本集
        y (np.array): 分类结果

    Returns:
        np.array: 投影方向
        np.array: 正例均值向量
        np.array: 反例均值向量
    """
    index_p = y == 1  # 正例
    index_n = y == 0  # 反例
    X_p = X[index_p]
    X_n = X[index_n]
    u_p = np.mean(X_p, axis=0).reshape(1, -1)
    u_n = np.mean(X_n, axis=0).reshape(1, -1)
    sigma_p = np.cov(X_p.T) * (X_p.shape[0] - 1)
    sigma_n = np.cov(X_n.T) * (X_n.shape[0] - 1)
    sigma_w = sigma_p + sigma_n
    w = np.linalg.inv(sigma_w) @ (u_n - u_p).T

    return w, u_p, u_n


def predict(X, w, u0, u1):
    project = np.dot(X, w)

    wu0 = np.dot(w.T, u0.T)
    wu1 = np.dot(w.T, u1.T)

    return (np.abs(project - wu1) < np.abs(project - wu0)).astype(int)


if __name__ == '__main__':

    data_path = './dataset/watermelon3_0_Ch.csv'
    # dataframe转为np.array格式
    data = pd.read_csv(data_path).values

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    w, u1, u0 = LDA_model(X,  y)
    print(f'训练集分类：{y.reshape(1, -1)}')
    print(f'LDA分类：{predict(X, w, u0=u0, u1=u1).reshape(1, -1)}')
