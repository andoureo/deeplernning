# coding:utf-8
import cv2
import keras
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = load_model('model/keras-mnist-model.h5')

def list_pictures(directory, ext='jpga|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

for picture in list_pictures('gazou/'):
        X = []
        img = img_to_array(load_img(picture, target_size=(28, 28), grayscale=True))
        X.append(img)

        X = np.asarray(X)
        X = X.astype('float32')
        X = X / 255.0

        features = model.predict(X)

        print('----------')
        print(picture)
        print(features.argmax())
        print('----------')
