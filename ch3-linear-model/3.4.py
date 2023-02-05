# 比较10折交叉验证法和留一法所估计出的对数几率回归的错误率
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score


def cal_score_using_sklearn(X, y):
    """
    使用sklearn自带的模型计算

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
    """

    sample = X.shape[0]
    print('使用sklearn模型：')
    # k-10 cross validation
    lr = linear_model.LogisticRegression(C=2)
    score = cross_val_score(lr, X, y, cv=10)
    print(f'10折交叉验证的平均分：{score.mean()}')

    # LOO
    loo = LeaveOneOut()
    accuracy = 0
    for train, test in loo.split(X, y):
        lr_ = linear_model.LogisticRegression(C=2)
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        lr_.fit(X_train, y_train)
        accuracy += lr_.score(X_test, y_test)
    print(f'留一法的平均分：{accuracy / sample}')


def cal_score_using_my_model(X, y):
    """
    使用自定义的模型计算

    Args:
        X (np.array): 样本集 (样本数 × 特征维度数)
        y (np.array): 分类结果 (样本数 × 1)
    """

    sample = X.shape[0]
    print('使用自定义模型：')
    # k-10
    # 这里只使用了748个样本中的740个。
    num_split = int(sample / 10)
    score_ten = []
    for i in range(10):
        lr = linear_model.LogisticRegression(C=2)
        test_index = range(i * num_split, (i + 1) * num_split)
        X_test = X[test_index]
        y_test = y[test_index]
        X_train = np.delete(X, test_index, axis=0)
        y_train = np.delete(y, test_index, axis=0)
        lr.fit(X_train, y_train)
        score_ten.append(lr.score(X_test, y_test))
    print(f'10折交叉验证的平均分：{np.mean(score_ten)}')

    # LOO
    score_loo = []
    for i in range(sample):
        lr = linear_model.LogisticRegression(C=2)
        X_test = X[i, :]
        y_test = y[i]
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        lr.fit(X_train, y_train)
        score_loo.append(int(lr.predict(X_test.reshape(1, -1)) == y_test))
    print(f'留一法的平均分：{np.mean(score_loo)}')


if __name__ == '__main__':
    data_path = './dataset/transfusion.data'
    data = np.loadtxt(data_path, delimiter=',', dtype=str)

    X = data[1:, :4].astype(int)
    y = data[1:, 4].astype(int)
    sample, feature = X.shape

    # 标准化
    X = (X - X.mean(0)) / X.std(0)
    # 打乱顺序
    index = np.arange(sample)
    np.random.shuffle(index)
    X = X[index]
    y = y[index]

    cal_score_using_sklearn(X, y)
    cal_score_using_my_model(X, y)
