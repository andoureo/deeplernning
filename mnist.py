import cv2
import keras
import numpy as np
import collections
import os
import re
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                                    # おまじない的な
model = load_model('model/keras-mnist-model.h5')                            # モデル読み込み

def remake(frame):
    threshold = 100
    ret, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)    # 二値化
    frame = cv2.bitwise_not(frame)                                          # BW反転
    img = cv2.resize(frame[150:350,200:400],(28,28))                        # 指定座標を28×28に変換
    return img,frame

def check_number(frame):
    X = []
    img = img_to_array(frame)                                               # 3次元の ndarrayに変換
    X.append(img)                                                           # 配列に加える
    X = np.asarray(X)                                                       # np配列に変換
    X = X.astype('float32')                                                 # タイプ変換
    X = X / 255.0                                                           # 0~255を0~1の範囲に収めるため
    num = model.predict(X)                                                  # ここで判別
    tmp = str(num.argmax())                                                 # 簡単に説明すると0-9ずつ確率をそれぞれ出し(合計100%)一番高い値をここで取り出す
    return tmp

def main():
    cap = cv2.VideoCapture(0)
    ans = []
    text = "-"
    while(cap.isOpened()):
        ret,frame = cap.read()
        frame1 = frame
        frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)             # グレイスケール化
        img,frame = remake(frame)
        ansnum = check_number(img)
        ans.append(ansnum)

        if (len(ans)>10):
            c = collections.Counter(ans)
            text = c.most_common()[0][0]
            ans.clear()

        cv2.putText(cv2.rectangle(frame1, (200,   150), (400,  350), (255, 0, 0), 1, 4), text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow("num", frame1)
        cv2.imshow("BW",frame)

        key = cv2.waitKey(10)
        if key == 27:                                                       # escキーを押したら終了
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
