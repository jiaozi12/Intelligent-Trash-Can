# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:07:10 2020

@author: qiqi

主函数

"""

import os
import cv2
import time
import joblib
import socket
import threading
import subprocess
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO
from Ultrasonic import ultrasonic
from UI import showUI, information
from _XiaoRGEEK_SERVO_ import XR_Servo
from Image_preprocessing import preprocess


'''类别信息'''
class_small = ['电池', '瓦砾陶瓷', '易拉罐', '烟头', '药品', '水果', '玻璃瓶', '办公用纸', 
               '纸质饮料盒', '塑料包装', '塑料水瓶', '卫生纸', '牙刷', '牙膏', '蔬菜']
class_large = {'电池':'有害垃圾', '瓦砾陶瓷':'其他垃圾', '易拉罐':'可回收垃圾',
               '烟头':'其他垃圾', '药品':'有害垃圾', '水果':'厨余垃圾',
               '玻璃瓶':'可回收垃圾', '办公用纸':'可回收垃圾', '纸质饮料盒':'可回收垃圾',
               '塑料包装':'其他垃圾', '塑料水瓶':'可回收垃圾', '卫生纸':'其他垃圾', 
               '牙刷':'其他垃圾', '牙膏':'其他垃圾', '蔬菜':'厨余垃圾'}

def read_img_s():
    while cap.grab():
        pass

def full_load_detection(dis):
    full = ''
    dis_limit = {'可回收垃圾':11, '有害垃圾':11, '其他垃圾':11, '厨余垃圾':11}
    for key in list(dis_limit.keys()):
        if dis[key] < dis_limit[key]:
            full += key + ' '
    if full != '':
        full += '  满载'
    else:
        full = '未满载'
    return full

tflite_model_quant_file = 'models/MobileNetV1.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_quant_file)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
XGB = joblib.load('models/XGBoost.pkl')
print('模型加载完毕')
mask = cv2.imread('mask.jpg')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#可回收垃圾（左上角），有害垃圾（右上角），其他垃圾（右下角），厨余垃圾（左下角）
pin = {'可回收垃圾':(17, 4), '有害垃圾':(14, 15), '其他垃圾':(27, 18),
       '厨余垃圾':(7, 5), '是否有垃圾放置':(24, 10)}

theta = {'可回收垃圾':130, '有害垃圾':82, '其他垃圾':35, '厨余垃圾':172}
wait_time = {'可回收垃圾':1, '有害垃圾':0, '其他垃圾':1, '厨余垃圾':2}

for TRIG, ECHO in list(pin.values()):
    GPIO.setup(TRIG,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ECHO,GPIO.IN, pull_up_down=GPIO.PUD_UP)

Servo = XR_Servo()
Servo.XiaoRGEEK_SetServoAngle(2, 82)  #2号舵机转到82度（右上角）
Servo.XiaoRGEEK_SetServoAngle(1, 25)  #1号舵机回到初始位置
print('舵机初始化完成')
cap = cv2.VideoCapture(0)
success, img = cap.read()
read_img = threading.Thread(target=read_img_s)
read_img.start()
print('Opencv线程启动')
showui = threading.Thread(target=showUI)
showui.start()
print('开始进入主循环工作')
lock = threading.Lock()  #超声波锁

#链接服务器ip和端口，并生成一个socket对象
ip_port = ('121.4.201.50', 9999)
sk = socket.socket()
sk.setblocking(False)

while success:
    lock.acquire()
    dis = ultrasonic(pin)
    lock.release()
    full = full_load_detection(dis)
    
    if dis['是否有垃圾放置'] >= 14:
        _, img = cap.read()
        img = cv2.resize(img, (224, 224))
        b, g, r = cv2.split(img)
        img_show = cv2.merge([r, g, b])
        information['摄像头画面帧'] = img_show
        information['任务是否完成'] = '当前无分类任务'
        information['垃圾类别'] = '无'
        information['窗口'] = '垃圾分类'
        information['满载情况'] = full
        continue
    time.sleep(1)

    success, img = cap.read() #从摄像头中读取图像
    
    '''预处理'''
    start = time.time()
    trash, source, img = preprocess(img, mask)
    
    end = time.time()

    if source is None:
        continue

    img = img.astype(np.uint8)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    information['摄像头画面帧'] = img
    information['任务是否完成'] = '未完成'
    information['窗口'] = '垃圾分类'
    information['满载情况'] = full

    print('开始上传图片')
    #将图片img上传至服务器
    try:
        i = cv2.resize(img, (224, 224))
        sk.connect(ip_port)
        sk.send(i.tobytes())
        sk.close()
    except:
        print('上传图片失败!!!')

    print('预处理花费时间:',round(end-start, 2),  's')
    start = time.time()
    
    img = img.astype(np.float32)
    img = img * 1./255 #归一化
    img = np.expand_dims(img, axis=0) #添加batch维度
    
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    p1 = interpreter.get_tensor(output_index)
    y = XGB.predict(p1)[0]
    result = class_small[y] #分类预测
    
    end = time.time()
    print('MobileNet预测花费时间:', round(end-start, 2), 's', '    预测类别:', result)
    information['垃圾类别'] = class_large[result]
    
    if full != '未满载' and class_large[result] in full:
        time.sleep(3)
        
    else:
        Servo.XiaoRGEEK_SetServoAngle(2, theta[class_large[result]])
        time.sleep(wait_time[class_large[result]])
        Servo.XiaoRGEEK_SetServoAngle(1, 125)
        time.sleep(2)
        Servo.XiaoRGEEK_SetServoAngle(1, 25)
        information['任务是否完成'] =  '任务完成'
        time.sleep(2)
        Servo.XiaoRGEEK_SetServoAngle(2, 82)
        time.sleep(wait_time[class_large[result]] + 1)
    
cap.release()