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

def remake(frame):
    threshold = 100
    ret, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    frame = cv2.bitwise_not(frame)
    img = cv2.resize(frame[150:350,200:400],(28,28))
    return img,frame

def check_number(frame):
    X = []
    img = img_to_array(frame)
    X.append(img)
    X = np.asarray(X)
    X = X.astype('float32')
    X = X / 255.0
    num = model.predict(X)
    ans = str(num.argmax())
    return ans

def main():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret,frame = cap.read()
        frame1 = frame
        frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        img,frame = remake(frame)
        ansnum = check_number(img)

        cv2.putText(cv2.rectangle(frame1, (200,   150), (400,  350), (255, 0, 0), 1, 4), ansnum, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow("num", frame1)
        cv2.imshow("二値化",frame)

        key = cv2.waitKey(10)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
