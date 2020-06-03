'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# データを128個ずつに分けて60000(70000?)/128回学習を行う
batch_size = 128
# 0~9
num_classes = 10
# 60000(70000?)/128回を12回行う
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# モデル層を積み重ねる形式の記述方法
model = Sequential()
# 3*3のフィルタを32枚使用,ReLU関数を使用,(1,28,28)or(28,28,1)
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 3*3のフィルタを64枚使用,ReLU関数を使用
model.add(Conv2D(64, (3, 3), activation='relu'))
# 2*2分のプーリング,2*2の範囲内の中から最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
# 全結合層とのつながりを25%無効,過学習予防
model.add(Dropout(0.25))
# 1次元ベクトルに変換
model.add(Flatten())
# 全結合層,出力128,ReLU関数
model.add(Dense(128, activation='relu'))
# 全結合層とのつながりを50%無効
model.add(Dropout(0.5))
# 全結合層,出力10,softmax関数
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
