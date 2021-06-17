# -*- coding: utf-8 -*-

import tensorflow_datasets as tfds
from matplotlib import pylab
from matplotlib import pyplot as plt
import numpy as np

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

def plot_imgs(imgs):
    plt.figure(dpi=300)
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        pylab.imshow(imgs[i])

train = tfds.load('mnist', split='train', shuffle_files=True)
train = train.shuffle(1024).batch(128).repeat(5).prefetch(10)
for example in tfds.as_numpy(train):
    imgs = data_pre_processing(example['image'])
    plot_imgs(imgs)
    break