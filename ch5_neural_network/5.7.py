import time

import matplotlib.pyplot as plt
import numpy as np

from __init__ import *
from ch5_neural_network.rbf import RBF


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.close('all')

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([[0], [1], [1], [0]])

    hidden = 10
    max_iter = 500
    c = X[np.random.randint(0, X.shape[0], size=hidden)]  # 随机采样确定中心点

    rbf_model = RBF()
    rbf_model.init(hidden=hidden, c=c)

    st = time.time()
    loss = rbf_model.train(X, y, eta=0.05, n=max_iter)
    ed = time.time()
    probability = rbf_model.predict(X)
    print(probability)
    y_predict = [1 if _p[0] > 0.5 else 0 for _p in probability]
    error = 0
    for _index, _y in enumerate(y.reshape(y.shape[0])):
        if _y != y_predict[_index]:
            error += 1
    print(f'RBF网络正确率：{round(1-error/len(y), 2)}, 用时：{round(ed-st, 2)}s')

    # Loss可视化
    plt.figure()
    plt.plot([i+1 for i in range(max_iter)], loss)
    plt.legend(['RBF'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("rbf_loss.png")
