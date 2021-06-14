from math import exp

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image,isRaw=0):
    """
    :param image:二维图片
    :param isRaw:原始数据集4维时候设为1,使用2维时候为0
    """
    ax = plt.subplot(5, 2, 1)
    if(isRaw!=0):
        ax.imshow(image, cmap='binary')
    else:
        N=image.shape[0]
        ax.imshow(image.reshape(N,N),cmap='binary')
    plt.show()

def im2col_forward(orign, core_w, core_h, stride, pad=0):  # 将矩阵按卷积核拉伸成向量
    '''
    :param orign:单个神经元，即二维矩阵
    :param core_h:卷积核的长度
    :param core_w:卷积核的高度
    :param stride:步长
    :param pad:是否零填充，默认否
    :return result:将卷积核拉伸成的向量矩阵，每一列代表卷积核匹配一次的值
    '''
    H, W = orign.shape
    out_h = int((H - core_h + pad * 2) / stride + 1)  # 纵向上能匹配卷积核几次
    out_w = int((W - core_w + pad * 2) / stride + 1)  # 横向上能匹配卷积核几次
    pad_image = np.pad(orign, ( (pad, pad), (pad, pad)), 'constant')
    result = np.zeros((out_h * out_w, core_w * core_h))  # 每张图片拉伸之后的结果应该是如下矩阵：(匹配的卷积核次数，一个卷积核中的元素个数)
    for w in range(out_w):
        max_w = w * stride + core_w
        for h in range(out_h):  # 在矩阵上移动卷积核
            max_h = h * stride + core_h
            i = 0
            for x in range(w * stride, max_w):
                for y in range(h * stride, max_h):  # 将卷积核内的小矩阵拉伸成一维向量
                    result[w * h, i] = pad_image[x, y]
                    i += 1
    return result


'''
def im2col_backward(col, X_shape, core_w, core_h, stride, pad = 0):
    W, H = X_shape
    result = np.zeros((W, H))        
    return result
'''


def sigmoid(X):
    '''
    用于中间卷积的激活函数
    :param X:单个列向量
    :return X:该列向量计算sigmoid后的列向量，shape不变
    '''
    for i in range(X.shape[0]):
        X[i][0] = 1 / (1 + exp(-X[i][0]))
    return X

def softmax(X):
    '''
    用于最后一层全连接的激活函数
    :param X:单个行向量
    :return result:计算softmax后对应行向量的值，shape不变
    '''
    sum = 0.
    for i in range(X.shape[0]):
        sum += np.exp(X[i])
    result = []
    for i in range(X.shape[0]):
        result.append(np.exp(X[i]) / sum)
    return result


def convolution_forward(X, W, b, stride=1, pad=0):
    '''
    :param X:单张图片(经过im2col的一维向量)
    :param W:权值
    :param b:偏置
    :param stride:步长
    :param pad:是否零填充
    :return result:卷积后的图片矩阵（一维）
    '''
    result = np.dot(X, W)
    result += b  # WX+b
    result = sigmoid(result)
    return result

def Affine_forward(X, W, b):
    '''
    :param X: 单张图片(经过im2col的一维向量)
    :param W: 权值
    :param b: 偏置
    :return:  全连接后的图片矩阵（一维）
    '''
    result = np.dot(X, W)
    result += b  # WX+b
    return result

def pooling_forward(X, X_shape, pool_w, pool_h):
    '''
    :param X:单张图片(一维向量)
    :param X_shape: 原图像的shape，因为经过im2col无法获取原图像格式
    :param pool_w: 池化矩阵的宽度
    :param pool_h: 池化矩阵的高度
    :return: result:单张图片(二维矩阵)
    '''
    W, H = X_shape
    temp = np.zeros((W, H))
    nw = int(W / pool_w)
    nh = int(H / pool_h)  # 按池化将卷积后的矩阵按池化矩阵分割
    result = np.zeros((nw, nh))  # 池化后的矩阵大小
    for i in range(X.shape[0]):
        temp[int(i / W), i % W] = X[i][0]  # 首先把向量还原回矩阵结构
    write = 0
    for w in range(0, W, pool_w):
        for h in range(0, H, pool_h):  # 在大矩阵内移动池化矩阵
            max_w = w + pool_w
            max_h = h + pool_h
            aveg = 0
            for x in range(w, max_w):
                for y in range(h, max_h):  # 每个池化矩阵求其内所有元素的平均值
                    aveg += temp[x, y]
            aveg /= (pool_w * pool_h)
            result[int(write / nw), write % nw] = aveg
            write += 1
    return result

def forward(X, W, b):
    '''
    封装好单张图片的前向传播：原始图片(28*28)->im2col和卷积(24*24*3)->池化(12*12*3)->im2col和全连接(1*3)
    :param X: 传入的单张图片(2维图像)
    :param W: 权值矩阵(所有层用到的权值都要传进来)，同层的共享权值
    :param b: 同上权值
    :return: 3类结果的预测值，1*3的向量。
    '''
    cal_X = im2col_forward(X, 5, 5, 1)  # 首先把矩阵拉伸成向量
    conv_result = np.zeros((3,576,1))
    pooling_result = np.zeros((3, 12, 12))  # 预设超参数，3个5*5的卷积核，池化矩阵为2*2
    result = np.zeros((3))
    for i in range(3):
        conv_result[i] = convolution_forward(cal_X, W[0][i], b[0][i]) #卷积层
        pooling_result[i] = pooling_forward(conv_result[i], [24, 24], 2, 2) #池化层
        # 全连接（注意池化出来的是二维，因此需要换为一维再计算全连接）
        result[i] = Affine_forward(im2col_forward(pooling_result[i], 12, 12, 1), W[1][i], b[1][i])
    result = softmax(result)
    #print(result)
    return result

def forward_all(images,W,b):
    '''
    :param images:所有图片
    :param W: 权值矩阵
    :param b: 偏置矩阵
    :return: 所有图片的预测值
    '''
    result=[]
    for image in images:
        result.append(forward(image,W,b))
    return result

def data_pre_processing(images):
    '''
    降维操作
    :param images:原始数据
    :return: 去掉最后一维颜色深度的数据
    '''
    N, H, W, C = images.shape
    down_images = np.zeros((N, H, W))
    for n in range(N):
        for x in range(H):
            for y in range(W):
                down_images[n][x][y] = images[n][x][y][0]
    return down_images

def weight_initialise():
    '''
    初始化权值
    '''
    #卷积层
    W1 = np.ones((3, 5 * 5, 1))
    b1 = np.zeros((3))
    #全连接层
    W2 = np.ones((3 , 12*12, 1))
    b2 = np.zeros((3))
    #组合
    W=[W1,W2]
    b=[b1,b2]
    return W,b

if __name__ == "__main__":
    train = tfds.load('mnist', split='train', shuffle_files=True)
    train = train.shuffle(1024).batch(128).repeat(5).prefetch(10)
    for example in tfds.as_numpy(train):
        data, labels = example["image"], example["label"]
    images = data_pre_processing(data)
    plot_image(images[0])
    W,b=weight_initialise()
    result = forward_all(images, W, b)  # 选一张图片进行计算
    print(result)
