import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import glob
import pandas as pd
import flask
from flask import request, jsonify, Response

model = tf.keras.models.load_model('f.h5')
cap = cv2.VideoCapture("dosya.mp4")
harfler = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
           "w", "x", "y"]
train = pd.read_csv("D:\\keras_p\\sign_mnist_train.csv")
test = pd.read_csv("D:\\keras_p\\sign_mnist_test.csv")

x = []

labels = train['label'].values
test_label = test['label']
unique_val = np.array(labels)
np.unique(unique_val)

f = open("kelimeler.txt", "w")


def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if (img_array is None):
        print("hata")
    else:
        new_array = cv2.resize(img_array, (28, 28))
        return new_array.reshape(-1, 28, 28, 1)


def getFrame(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = cap.read()
    roi = image[20:110, 50:140]

    if hasFrames:
        cv2.imwrite("image" + str(count) + ".jpg", roi)
        prediction = model.predict([prepare("D:\\udemy-dersleri\\proje\\deneme\\image"+str(count)+".jpg")])

        # print(prediction[0])
        """
        if prediction[0][0] == 1:
            print(harfler[0])
        if prediction[0][5] == 1:
            print(harfler[1])
        """

        for i in range(len(prediction)):
            for j in range(0, 25):
                if prediction[i][j] == 1:
                    #print(harfler[j])
                    x.append(harfler[j])


        a = ''.join(x)
        print("--------")
        print(a)
        f = open("kelimeler.txt", "w")
        f.write("%s" % a)



    return hasFrames

sec = 0
frameRate = 1.2
count = 1
success = getFrame(sec)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)

    success = getFrame(sec)
    f.close()


