# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:04:22 2021

@author: qiqi
"""

import os
import joblib 
import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model

small_class = {0:'Battery', 1:'Brick_And_Tile', 2:'Can', 3:'Cigarette_Butt', 
               4:'Drugs', 5:'Fruits', 6:'Glass_Bottle', 7:'Paper', 8:'Paper_Drink_Box',
               9:'Plastic_Bag', 10:'Plastic_Bottle', 11:'Toilet_Paper', 12:'Toothbrush',
               13:'Toothpaste', 14:'Vegetables'}

large_class = {'Battery':'可回收垃圾', 'Brick_And_Tile':'其他垃圾', 'Can':'可回收垃圾',
               'Cigarette_Butt':'其他垃圾', 'Drugs':'有害垃圾', 'Fruits':'厨余垃圾',
               'Glass_Bottle':'可回收垃圾', 'Paper':'可回收垃圾', 
               'Paper_Drink_Box':'可回收垃圾', 'Plastic_Bag':'其他垃圾',
               'Plastic_Bottle':'可回收垃圾', 'Toilet_Paper':'其他垃圾',
               'Toothbrush':'其他垃圾', 'Toothpaste':'其他垃圾', 'Vegetables':'蔬菜'}


def Acc(X, y, clf):
    p = clf.predict(X)
    right = 0
    right_large = 0
    for i in range(len(p)):
        if p[i] == y[i]: right += 1
        if large_class[small_class[p[i]]] == large_class[small_class[y[i]]]: right_large += 1
    return right / len(p), right_large / len(p)

def train_and_test(train_root, test_root, model):
    #训练并评价MobileNetV1+XGBoost算法精度
    classes = os.listdir(train_root)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for cla in classes:
        y = classes.index(cla)
        files = os.listdir(train_root + '/' + cla)
        for file in files:
            path = train_root + '/' + cla + '/' + file
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img) / 255
            X_train.append(x)
            y_train.append(y)
        files = os.listdir(test_root + '/' + cla)
        for file in files:
            path = test_root + '/' + cla + '/' + file
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img) / 255
            X_test.append(x)
            y_test.append(y)
    
    X_train = model.predict(np.array(X_train))
    y_train = np.array(y_train)
    X_test = model.predict(np.array(X_test))
    y_test = np.array(y_test)
    state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)
    XGB = XGBClassifier(booster='gbtree', subsample=0.75, colsample_bytree=0.75, learning_rate=0.1)
    XGB.fit(X_train, y_train)
    joblib.dump(XGB, 'models/XGBoost.pkl')
    print('MobileNetV1+XGBoost Acc:', Acc(X_test, y_test, XGB))
    
if __name__ == '__main__':
    base_model = load_model('models/MobileNetV1.h5')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)
    train_and_test('data_trash/train', 'data_trash/test', model)