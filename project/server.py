
from flask import Flask, render_template, request

from scipy.misc import imsave, imread, imresize

import numpy as np
import base64

import keras.models

import re


import sys

import os
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

from keras.models import load_model
import tensorflow as tf


sys.path.append(os.path.abspath("./model"))
from load import *


app = Flask(__name__)


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('fruits_better.h5')


print("Loaded Model from disk")

#compile and evaluate loaded model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

graph = tf.get_default_graph()


def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('output.png','wb') as output:
      output.write(base64.b64decode(imgstr))


@app.route('/')
def index():

    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():

    imgData = request.get_data()

    convertImage(imgData)
    print("debug")

    x = imread('output.png', mode='L')

    x = np.invert(x)

    x = imresize(x, (28, 28))

    x = x.reshape(1, 28, 28, 1)
    print("debug2")

    with graph.as_default():

        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print("debug3")

        FRUITS = ["Pear","Grapes", "Strawberry","Blueberry", "Pineapple", "banana", "apple","Watermelon"]

        print(out)
        print(type(np.argmax(out, axis=1)))
        stick = np.argmax(out, axis=1)
        s1 = list(stick.astype(int))
        print(s1)


        for i in s1:
            print(i)

        response=FRUITS[i]
        return response


if __name__ == "__main__":

    port = int(os.environ.get('PORT', 5500))

    app.run(host='0.0.0.0', port=port)

app.run(debug=True)