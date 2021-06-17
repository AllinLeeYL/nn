"""
优化器
每一个优化器函数都返回对应函数名的导函数值
"""
from copy import copy

def __affine_sum_b(W, c, i, n):
    total = 0
    for j in range(W[1].shape[1]):
        w = j % n
        h = int(j / n)
        l1 = 2*w + 2*h*2*n
        l2 = 2*w+1 + 2*h*2*n
        l3 = 2*w + (2*h+1)*2*n
        l4 = 2*w+1 + (2*h+1)*2*n
        total += c[i, l1]*(1-c[i, l1])
        total += c[i, l2]*(1-c[i, l2])
        total += c[i, l3]*(1-c[i, l3])
        total += c[i, l4]*(1-c[i, l4])
    return total

def __affine_sum_w(W, c, u, i, k, n):
    total = 0
    for j in range(W[1].shape[1]):
        w = j % n
        h = int(j / n)
        l1 = 2*w + 2*h*2*n
        l2 = 2*w+1 + 2*h*2*n
        l3 = 2*w + (2*h+1)*2*n
        l4 = 2*w+1 + (2*h+1)*2*n
        total += c[i, l1]*(1-c[i, l1]) * u[l1, k]
        total += c[i, l2]*(1-c[i, l2]) * u[l2, k]
        total += c[i, l3]*(1-c[i, l3]) * u[l3, k]
        total += c[i, l4]*(1-c[i, l4]) * u[l4, k]
    return total

def bp(W, b, params, alpha):
    y_res = params[0] # 标准输出 numpy.array[3]
    u = params[1]  # 卷积层的输入矩阵 numpy.array[3, 576, 25]
    c = params[2]  # sigmoid激活函数输出 numpy.array[3, 576]
    n = params[3]  # 池化层输出的宽或高
    a = params[4]  # 全连接层输入 numpy.array[3, 144]
    y_p = params[5]  # 预测输出 numpy.array[3]
    W_old = copy(W)
    # 梯度下降
    for i in range(W[0].shape[0]):
        t = (y_p[i] - y_res[i]) * alpha
        # 全连接层参数-偏置b
        b[1][i] -= t
        # 全连接层参数-权值w
        for j in range(W[1].shape[1]):
            W[1][i, j] -= t*a[i, j]
        # 卷积层参数-偏置b
        total = __affine_sum_b(W, c, i, n)
        b[0][i] -= t * W_old[1][i, j] * total
        # 卷积层参数-权值w
        for k in range(W[0].shape[1]):
            total = __affine_sum_w(W, c, u, i, k, n)
            W[0][i, k] -= t * W_old[1][i, j] * total
    return W, b