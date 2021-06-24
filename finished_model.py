# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:39:27 2021
@author: chomi
"""

import tensorflow.keras
from PIL import Image, ImageOps
from detecto import core
import numpy as np
import os

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# importing tranined moddels
model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)
cdmodel = core.Model.load('model.pth', ['sample'])


# taking prediction and classifying volume
def classify(prediction, filename):

    print("Classifying %s, prediction:" % filename, prediction)

    x = filename.split(".png")
    if (prediction[0][0] > 0.8):
        print(x[0], "volume close to ZERO")
        zero.append(filename)
        z.append(prediction[0][0])

    elif (prediction[0][1] > 0.8):
        print(x[0], "volume is ONE")
        one.append(filename)
        o.append(prediction[0][1])

    elif (prediction[0][2] > 0.8):
        print(x[0], "volume is TWO")
        two.append(filename)
        t.append(prediction[0][2])

    elif (prediction[0][3] > 0.8):
        print(x[0], "volume is THREE")
        three.append(filename)
        th.append(prediction[0][3])

    elif (prediction[0][4] > 0.8):
        print(x[0], "volume is FOUR")
        four.append(filename)
        f.append(prediction[0][4])

    else:
        idx = np.where(prediction[0] == np.amax(prediction[0]))
        prediction[0][idx[0][0]] += 0.8
        print("Unclear but most likely ", end='')
        classify(prediction, filename)


# arrays keep track of class and their confidence values
zero = []
one = []
two = []
three = []
four = []

z = []
o = []
t = []
th = []
f = []

path = "C:/Users/sam.cross/PycharmProjects/image_recognition/test_images/"  # path to the images
counter = 0
for filename in os.listdir(path):
    if (filename.endswith(".png")):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # crucible detection
        img = Image.open(path + filename)

        print("Analyzing image at", path+filename)

        pred = cdmodel.predict(img)
        lbl, box, score = pred
        temp = box.numpy()[0]
        box = (round(temp[0]), round(temp[1]), round(temp[2]), round(temp[3]))
        image = img.crop(box)  # passing this to powder model

        # powder stuff
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)

        # display image (to user) that model took as input
        # image.show()
        # run image through the model
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        classify(prediction, filename)
        counter += 1
        print(counter)

# writing the image names to separate .txt files
A = open("0.txt", "w+")
for i in range(len(zero)):
    A.write(zero[i] + ": " + str(z[i]) + '\n')
A.close()
B = open("1.txt", "w+")
for i in range(len(one)):
    B.write(one[i] + ": " + str(o[i]) + '\n')
B.close()
C = open("2.txt", "w+")
for i in range(len(two)):
    C.write(two[i] + ": " + str(t[i]) + '\n')
C.close()
D = open("3.txt", "w+")
for i in range(len(three)):
    D.write(three[i] + ": " + str(th[i]) + '\n')
D.close()
E = open("4.txt", "w+")
for i in range(len(four)):
    E.write(four[i] + ": " + str(f[i]) + '\n')
E.close()