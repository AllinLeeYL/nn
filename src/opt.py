"""
优化器
每一个优化器函数都返回对应函数名的导函数值
"""

def bp(W, b, params, alpha):
    y_res = params[0] # 标准输出 numpy.array[3]
    y_p = params[1] # 预测输出 numpy.array[3]
    a = params[2] # 全连接层输入 numpy.array[3, 144]
    c = params[3] # sigmoid激活函数输出 numpy.array[3, 576]
    u = params[4] # 卷积层的输入矩阵 numpy.array[3, 576, 25]
    n = params[5] # 池化层输出的宽或高
    # 梯度下降
    for i in range(len(W[0].shape[0])):
        t = (y_p[i] - y_res[i]) * alpha
        # 全连接层参数-偏置b
        b[1][i] -= t
        # 全连接层参数-权值w
        for j in range(len(W[1].shape[1])):
            W[1][i, j] -= t*a[i, j]
        # 卷积层参数-偏置b
        total = 0
        for j in range(len(W[1].shape[1])):
            w = j % n
            h = int(j / n)
            l1 = 2*w + 2*h*2*n
            l2 = 2*w+1 + 2*h*2*n
            l3 = 2*w + (2*h+1)*2*n
            l4 = 2*w+1 + (2*h+1)*2*n
            total += c[i, l1](1-c[i, l1])
            total += c[i, l2](1-c[i, l2])
            total += c[i, l3](1-c[i, l3])
            total += c[i, l4](1-c[i, l4])
        b[0][i] -= t * W[1][i, j] * total
        # 卷积层参数-权值w
        for k in range(len(W[0].shape[1])):
            total = 0
            for j in range(len(W[1].shape[1])):
                w = j % n
                h = int(j / n)
                l1 = 2*w + 2*h*2*n
                l2 = 2*w+1 + 2*h*2*n
                l3 = 2*w + (2*h+1)*2*n
                l4 = 2*w+1 + (2*h+1)*2*n
                total += c[i, l1](1-c[i, l1]) * u[i, l1, k]
                total += c[i, l2](1-c[i, l2]) * u[i, l2, k]
                total += c[i, l3](1-c[i, l3]) * u[i, l3, k]
                total += c[i, l4](1-c[i, l4]) * u[i, l4, k]
            W[0][i, k] -= t * W[1][i, j] * total
    return W, b