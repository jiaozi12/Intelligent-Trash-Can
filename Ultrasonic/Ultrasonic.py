# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:33:42 2020

@author: qiqi

此文件用于实现四分类垃圾桶的5个超声波测距的控制

"""

import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def getDistance(TRIG=None, ECHO=None):
    '''函数功能：获取ECHO与TRIG指定的超声波传感器测定的距离'''
    '''TRIG为超声波发射脚位，ECHO为超声波接收脚位'''
    if TRIG is None or ECHO is None:
        print('引脚未接好')
        return 0
    GPIO.output(TRIG,GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(TRIG,GPIO.LOW)
    while not GPIO.input(ECHO):
        pass
    t1 = time.time()
    while GPIO.input(ECHO):
        pass
    t2 = time.time()
    Distence = (t2-t1)*340/2*100
    
    return Distence

def ultrasonic(pin):
    dis = pin.copy()
    key = list(pin.keys())
    for k in key:
        dis[k] = getDistance(TRIG=pin[k][0], ECHO=pin[k][1])
    return dis