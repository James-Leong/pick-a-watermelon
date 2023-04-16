# 线性可分样本集下，LDA与SVM的对比
from sklearn import svm, discriminant_analysis
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# 取出 0，1类作为训练样本
target = pd.Series(y).isin([0, 1])

# 绘制样本点
fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 2], c='cornflowerblue')  # 正例
ax.scatter(X[y == 1, 0], X[y == 1, 2], c='darkorange')  # 反例
# 坐标轴
x_tmp = np.linspace(4, 7.2, 2040)
y_tmp = np.linspace(0, 5.2, 2040)
X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

# LDA
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)
lda.fit(X[:, [0, 2]][target], y[target])
# 预测点
Z_lda = lda.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)
# 绘制LDA等高线
C_lda = ax.contour(X_tmp, Y_tmp, Z_lda, [0.5], colors='orange', linewidths=1)
plt.clabel(C_lda, fmt={C_lda.levels[0]: 'lda decision boundary'}, inline=True, fontsize=8)

# SVM
y[y == 0] = -1
linear_svm = svm.SVC(kernel='linear', C=10000)
linear_svm.fit(X[:, [0, 2]][target], y[target])
# 预测点
Z_svm = linear_svm.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)
# 绘制SVM等高线
C_svm = ax.contour(X_tmp, Y_tmp, Z_svm, [0], colors='g', linewidths=1)
plt.clabel(C_svm, fmt={C_svm.levels[0]: 'svm decision boundary'}, inline=True, fontsize=8,
           manual=[(6.2, 2.4)])
# 标记SVM支持向量
ax.scatter(linear_svm.support_vectors_[:, 0], linear_svm.support_vectors_[:, 1],
           marker='o', c='none', edgecolors='m', s=150)

plt.show()
