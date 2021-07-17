# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:59:15 2020

@author: qiqi

此文件用于实现四分类垃圾桶两个UI界面的显示工作

"""

import sys
import warnings
import threading
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer,QDateTime
warnings.filterwarnings('ignore')

information = {'垃圾类别':'电池', '任务是否完成':'任务完成', 
               '满载情况':'未满载', '摄像头画面帧':None, '窗口':None, 
               '宣传视频帧':None}

class showTrash(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        desktop = QApplication.desktop()
        self.label_0 = QtWidgets.QLabel(self) #用来显示垃圾类别
        self.label_0.resize(desktop.width()*0.4, desktop.height()*0.2)
        self.label_0.move(desktop.width()*0.1, desktop.height()*0.15)
        
        self.label_1 = QtWidgets.QLabel(self) #用来显示任务是否完成
        self.label_1.resize(desktop.width()*0.4, desktop.height()*0.2)
        self.label_1.move(desktop.width()*0.1, desktop.height()*0.35)
        
        self.label_2 = QtWidgets.QLabel(self) #用来显示满载情况
        self.label_2.resize(desktop.width()*0.4, desktop.height()*0.2)
        self.label_2.move(desktop.width()*0.1, desktop.height()*0.55)
        
        self.label_3 = QtWidgets.QLabel(self) #用来摄像头画面
        self.label_3.resize(desktop.width()*0.5, desktop.height()*0.5)
        self.label_3.move(desktop.width()*0.5, desktop.height()*0.15)
        
        self.setWindowTitle('垃圾桶工作状态')
        self.setGeometry(600, 600, 1000, 500)
        self.showMaximized()
        
    
    def setupUi(self):
        self.Timer=QTimer()     #自定义QTimer
        self.Timer.start(100)   #每0.1秒运行一次
        self.Timer.timeout.connect(self.update)   #连接update
        self.Timer.timeout.connect(self.showImage) 
        QtCore.QMetaObject.connectSlotsByName(self)
        
    def update(self):
        self.label_0.setText(information['垃圾类别'])
        self.label_0.setStyleSheet('color:rgb(10,10,10,255);font-size:20px;font-weight:bold;font-family:Roman times;')
        self.label_1.setText(information['任务是否完成'])
        self.label_1.setStyleSheet('color:rgb(10,10,10,255);font-size:20px;font-weight:bold;font-family:Roman times;')
        self.label_2.setText(information['满载情况'])
        self.label_2.setStyleSheet('color:rgb(10,10,10,255);font-size:20px;font-weight:bold;font-family:Roman times;')
    
    def showImage(self):
        frame = information['摄像头画面帧']
        heigt, width, _ = frame.shape
        pixmap = QImage(frame, width, heigt, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)
        self.label_3.setPixmap(pixmap)

class showVideo(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        desktop = QApplication.desktop()
        self.label = QtWidgets.QLabel(self)
        self.label.resize(desktop.width()*0.99, desktop.height()*0.79)
        self.label.move(desktop.width()*0.01, desktop.height()*0.01)
        self.setWindowTitle('宣传视频')
        self.showMaximized()
    
    def setupUi(self):
        self.Timer=QTimer()     #自定义QTimer
        self.Timer.start(100)   #每0.1秒运行一次
        self.Timer.timeout.connect(self.showImage) 
        QtCore.QMetaObject.connectSlotsByName(self)
    
    def showImage(self):
        frame = information['宣传视频帧']
        heigt, width, _ = frame.shape
        pixmap = QImage(frame, width, heigt, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)
        self.label.setPixmap(pixmap)
        if information['窗口'] == '垃圾分类':
            self.close()

class Thread(threading.Thread):
    def __init__(self, window, *args, **kwargs):
        super(Thread, self).__init__(*args, **kwargs)
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.window = window
    
    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
    
    def run(self):
        app = QApplication(sys.argv)
        if self.window == '视频播放':
            ui = showVideo()
        else:
            ui = showTrash()
        
        ui.setupUi()
        app.exec_()
    
def showUI():
    thread = Thread('垃圾分类')
    thread.start()
    information['窗口'] = '垃圾分类'
    thread.join()
