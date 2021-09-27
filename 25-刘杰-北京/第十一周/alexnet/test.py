#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：test.py
@Author  ：luigi
@Date    ：2021/9/16 5:58 下午 
'''

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import argparse
from imutils import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", required=True, help="specify the input path for test image set")
    ap.add_argument("-m", "--model_path", required=True, help="specify the input path for trained model")
    ap.add_argument("-b", "--batch_size", default=32, help="batch_size")
    args = vars(ap.parse_args())

    imagepaths = list(paths.list_images(args["image_path"]))
    for path in imagepaths:
        image = cv2.imread(path)
        image = cv2.resize(image, (227, 227))
        image = image.astype(np.float32) / 255
        image = np.expand_dims(image, axis=0)

        model = load_model(args["model_path"])

        classes = {0: 'cat', 1: "dog"}
        prediction = np.argmax(model.predict(image))
        print("path: {}, prediction: {}".format(path,classes[prediction]))


if __name__ == '__main__':
    main()
