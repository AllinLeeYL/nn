"""
优化器
每一个优化器函数都返回对应函数名的导函数值
"""

def bp(W, b, params, alpha):
    y_res = params[0] # 标准输出 numpy.array[3]
    y_p = params[1] # 预测输出 numpy.array[3]
    a = params[2] # 全连接层输入 numpy.array[144]
    c = params[3] # sigmoid激活函数输出 numpy.array[576]
    u = params[4] # 卷积层的输入矩阵 numpy.array[576, 25]
    # 梯度下降
    for i in range(len(W[0].shape[0])):
        t = (y_p[i] - y_res[i]) * alpha
        # 全连接层参数-偏置b
        b[1][i] -= t
        # 全连接层参数-权值w
        for j in range(len(W[1].shape[1])):
            W[1][i, j] -= t*a[j]
        # 卷积层参数-偏置b
        total = 0
        for j in range(len(W[1].shape[1])):
            w = j % 12
            h = int(j / 12)
            l1 = 2*w + 2*h*24
            l2 = 2*w+1 + 2*h*24
            l3 = 2*w + (2*h+1)*24
            l4 = 2*w+1 + (2*h+1)*24
            total += c[l1](1-c[l1])
            total += c[l2](1-c[l2])
            total += c[l3](1-c[l3])
            total += c[l4](1-c[l4])
        b[0][i] -= t * W[1][i, j] * total
        # 卷积层参数-权值w
        for k in range(len(W[0].shape[1])):
            total = 0
            for j in range(len(W[1].shape[1])):
                w = j % 12
                h = int(j / 12)
                l1 = 2*w + 2*h*24
                l2 = 2*w+1 + 2*h*24
                l3 = 2*w + (2*h+1)*24
                l4 = 2*w+1 + (2*h+1)*24
                total += c[l1](1-c[l1]) * u[l1, k]
                total += c[l2](1-c[l2]) * u[l2, k]
                total += c[l3](1-c[l3]) * u[l3, k]
                total += c[l4](1-c[l4]) * u[l4, k]
            W[0][i, k] -= t * W[1][i, j] * total
    return W, b