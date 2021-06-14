import numpy as np

"""
优化器
每一个优化器函数都返回对应函数名的导函数值
"""

# 均方误差函数的导数(真实结果，预测输出)
def MSE_bp(y, x):
    return y - x

# 交叉熵损失函数的导数(真实结果，预测输出)
def CE_bp(y, x):
    return (1-y)/(1-x) - y/x

# softmax函数的导数(向量X，求导的分量索引i)
def softmax_bp(X, i):
    total = sum([np.exp(x) for x in X])
    return (total - np.exp(X[i])) / (total**2)

# ([]，学习率)
# [真实输出，softmax输出，全连接输出，pooling输出，权值矩阵W, 卷积层输入X]
# [3], [3], [3], [1, 144], [2, ...]
def params_bp(y, x, X, pooling_result, W, C_X, alpha):
    W1 = np.ones((3, 5 * 5, 1))
    b1 = np.zeros((3))
    W2 = np.ones((3 , 12*12, 1))
    b2 = np.zeros((3))
    for i in range(0, 3):
        x1 = CE_bp(y[i], x[i]) # 交叉熵导数
        x1 *= softmax_bp(X, i) # softmax导数
        for j in range(0, 12*12):
            W2[i, j] = x1 * pooling_result[0, j] * alpha # 全连接导数
        b2[i] = x1 * alpha # 全连接导数
        for j in range(0, 12*12):
            x2 = x1 * W[1][i][j] # 全连接对x求导
            x2 *= pooling_result[0, j](1-pooling_result[0, j])
            w = j % 12
            h = j / 12
            for k in range(0, 5*5):
                W1[i, k] += x2 * X[2*w + 2*h*24, k]
                W1[i, k] += x2 * X[2*w+1 + 2*h*24, k]
                W1[i, k] += x2 * X[2*w + (2*h+1)*24, k]
                W1[i, k] += x2 * X[2*w+1 + (2*h+1)*24, k]
            b1[i] += x2
        W1[i, :] = W1[i, :] / (12 * 12) * alpha
        b1[i] = b1[i] / (12 * 12) * alpha
    return [W1, W2], [b1, b2]
    