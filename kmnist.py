# coding:utf-8
import cv2
import keras
import numpy as np
import collections
from PIL import ImageFont, ImageDraw, Image
from keras.models import model_from_json#load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img
import os
import re

label_dict = {"0":"あ", "1":"い", "2":"う", "3":"え", "4":"お", "5":"か", "6":"き", "7":"く", "8":"け", "9":"こ", "10":"さ", "11":"し", "12":"す", "13":"せ", "14":"そ", "15":"た", "16":"ち", "17":"つ", "18":"て", "19":"と", "20":"な", "21":"に", "22":"ぬ", "23":"ね", "24":"の", "25":"は", "26":"ひ", "27":"ふ", "28":"へ", "29":"ほ", "30":"ま", "31":"み", "32":"む", "33":"め", "34":"も", "35":"や", "36":"ゆ", "37":"よ", "38":"ら", "39":"り", "40":"る", "41":"れ", "42":"ろ", "43":"わ", "44":"ゐ", "45":"ゑ", "46":"を", "47":"ん", "48":"ゝ", "49":"-"}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model =model_from_json(open("model/k_mnist_cnn_model.json").read())
model.load_weights("model/k_mnist_cnn_weights.h5")

def remake(frame):
    threshold = 100
    ret, frame= cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    frame=cv2.bitwise_not(frame)
    img=cv2.resize(frame[150:350,200:400],(28,28))
    return img

def check_number(frame):
    X = []
    img = img_to_array(frame)
    X.append(img)
    X = np.asarray(X)
    X = X.astype("float32")
    X = X / 255.0
    num=model.predict(X)
    ans=str(num.argmax())
    return ans

def img_add_msg(img, message):
    font_path = 'C:\Windows\Fonts\meiryo.ttc'           # Windowsのフォントファイルへのパス
    font_size = 24                                      # フォントサイズ
    font = ImageFont.truetype(font_path, font_size)     # PILでフォントを定義
    img = Image.fromarray(img)                          # cv2(NumPy)型の画像をPIL型に変換
    draw = ImageDraw.Draw(img)                          # 描画用のDraw関数を用意
    # テキストを描画（位置、文章、フォント、文字色（BGR+α）を指定）
    draw.text((20, 50), message, font=font, fill=(255, 0, 0, 0))
    img = np.array(img)                                 # PIL型の画像をcv2(NumPy)型に変換
    return img

def main():
    cap = cv2.VideoCapture(0)
    ans = []
    text = "49"
    while(cap.isOpened()):
        ret,frame=cap.read()
        frame1=frame
        frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        img=remake(frame)
        ansnum=check_number(img)
        ans.append(ansnum)
        if (len(ans)>10):
            c = collections.Counter(ans)
            text = c.most_common()[0][0]
            ans.clear()
        tex=label_dict[text]

        frame1 = img_add_msg(frame1, tex)
        cv2.imshow("num", frame1)

        key = cv2.waitKey(10)
        if key == 27:
            #escキー
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
