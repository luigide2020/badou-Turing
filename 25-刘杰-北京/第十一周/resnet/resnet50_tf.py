#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：resnet50_tf.py
@Author  ：luigi
@Date    ：2021/9/27 11:56 上午 
'''

import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, add
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import argparse


def transform_image(image, label):
    x_train = tf.image.resize(image, (227, 227))
    x_train = tf.cast(x_train, tf.float32) / 255
    y_train = tf.one_hot(label, 2)
    return x_train, y_train


def get_data(image_path):
    train_set, val_set = tfds.load(name='cats_vs_dogs', split=["train[:75%]", "train[75%:]"],
                                   data_dir=image_path, shuffle_files=True, as_supervised=True)
    return train_set, val_set


def residual_model(input_tensor, K, chanDim, stride, reduce=False, reg=0.0001):

    shortcut = input_tensor

    # Block #1: first BN => RELU => CONV layer set
    bn1 = BatchNormalization(axis=chanDim)(input_tensor)
    act1 = Activation('relu')(bn1)
    conv1 = Conv2D(int(K * 0.25), (1, 1), padding='same', use_bias=False,
                   input_shape=input_tensor.shape, kernel_regularizer=l2(reg))(act1)

    # Block #2: second BN => RELU => CONV layer set
    bn2 = BatchNormalization(axis=chanDim)(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv2D(int(K * 0.25), (3, 3), padding='same', use_bias=False,kernel_regularizer=l2(reg))(act2)

    # Block #3: third BN => RELU => CONV layer set
    bn3 = BatchNormalization(axis=chanDim)(conv2)
    act3 = Activation('relu')(bn3)
    conv3 = Conv2D(K, (1, 1), padding='same', use_bias=False,kernel_regularizer=l2(reg))(act3)

    if reduce:
        shortcut = Conv2D(K, (1, 1), strides=stride,use_bias=False,kernel_regularizer=l2(reg))(act1)

    x = add(conv3, shortcut)

    return x


def get_model(reg=0.0002):
    model = Sequential()
