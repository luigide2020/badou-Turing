#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：alexnet_tf.py
@Author  ：luigi
@Date    ：2021/9/14 9:57 上午 
'''
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
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


def get_model(chanDim, reg=0.0002):
    model = Sequential()

    # Block #1: first CONV => RELU => POOL layer set
    model.add(
    Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=(227, 227, 3), kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block #2: second CONV => RELU => POOL layer set
    model.add(Conv2D(256, (5, 5), padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block #3: CONV => RELU => CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    # Block #4: first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(4096, kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Block #5: second set of FC => RELU layers
    model.add(Dense(4096, kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Block #6: second set of FC => RELU layers
    model.add(Dense(1000, kernel_regularizer=l2(reg)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(2, kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))

    return model


def train_batch(train, validation_data, model, epochs):
    opt = Adam(lr=1e-3)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train_x, train_y = train
    H = model.fit(train, validation_data=validation_data, epochs=epochs, batch_size=32)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label='train_loss')
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label='val_loss')
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label='accuracy')
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label='val_accuracy')
    plt.title("training loss and accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="specify the input path for cat_vs_dog")
    ap.add_argument("-o", "--output", required=True, help="specify the output path for trained model")
    ap.add_argument("-b", "--batch_size", default=32, help="batch_size")
    args = vars(ap.parse_args())

    train_set, valid_set = get_data(args["input"])
    train_set = train_set.map(transform_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_set = train_set.cache()
    train_set = train_set.batch(args["batch_size"])
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)

    valid_set = valid_set.map(transform_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.cache()
    valid_set = valid_set.batch(args["batch_size"])
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)

    model = get_model(chanDim=3)
    train_batch(train_set, valid_set, model, epochs=10)
    model.save(args["output"], overwrite=True)


if __name__ == '__main__':
    main()
