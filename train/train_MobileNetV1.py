# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:11:32 2020

@author: qiqi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:13:31 2020

@author: qiqi
"""

import os
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image

imageShape = (224, 224)
batch_size = 64
lenTrain = 2810  #训练集图片数量
lenVal = 929  #验证集图片数量
epochs = 300  #训练轮次
trainRoot = 'data_trash/train'
testRoot = 'data_trash/test'

train_data_gen = ImageDataGenerator(
    rotation_range=180, #整数。随机旋转的度数范围。
    width_shift_range=0.2, #浮点数、一维数组或整数
    height_shift_range=0.2, #浮点数。剪切强度（以弧度逆时针方向剪切角度）。
    shear_range=0.2, 
    zoom_range=0.2, #浮点数 或 [lower, upper]。随机缩放范围
    fill_mode='nearest', # {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充：
    cval=0.0, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rescale=1./255, 
    validation_split=0.25
)
train_batches = train_data_gen.flow_from_directory(
    trainRoot, target_size = imageShape, 
    batch_size = batch_size,
    class_mode = 'categorical',
    subset='training',
)
val_batches = train_data_gen.flow_from_directory(
    trainRoot, target_size = imageShape, 
    batch_size = batch_size,
    class_mode = 'categorical',
    subset='validation'
)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个分类器，假设我们有15个类
predictions = Dense(15, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.summary()

earlyStop = EarlyStopping(monitor='val_accuracy', 
                          patience=10, 
                          verbose=1, mode='auto', 
                          restore_best_weights=True)

_ = model.fit_generator(
    epochs=epochs, shuffle=True, callbacks=[earlyStop],
    validation_data=val_batches, generator=train_batches, 
    steps_per_epoch=lenTrain//batch_size, 
    validation_steps=lenVal//batch_size,verbose=1,
)
model.save('models/MobileNetV1.h5')
imgs = []
y = []
classes = os.listdir(testRoot)
for cla in classes:
    label = classes.index(cla)
    files = os.listdir(testRoot + '/' + cla)
    for file in files:
        path = testRoot + '/' + cla + '/' + file
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img) / 255
        imgs.append(x)
        y.append(label)

right = 0
p = model.predict(np.array(imgs))
for i in range(len(p)):
    if np.argmax(p[i]) == y[i]: right += 1
print('Acc:', round(right / len(y), 2))