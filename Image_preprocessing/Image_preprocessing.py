# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:01:53 2020

@author: qiqi

此文件用来实现图像预处理步骤

"""

import cv2
import math
import numpy as np

image_height = 224
image_width = 224

def unevenLightCompensate(img, blockSize):
    '''去除图像中的光照不均匀'''
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst[dst > 255] = 255
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (7, 7), 0)

    return dst

def edge_demo(image):
    '''使用Canny算法对image进行边缘检测，输入image为彩色图像'''
    edge_output = cv2.Canny(image, 50, 100)
    return edge_output

def rotate(img, box):
    '''旋转图像并剪裁'''
    pt1, pt2, pt3, pt4 = box
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    angle = math.acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度

    if pt4[1]>pt1[1]:
        info = '顺时针旋转'
    else:
        info = '逆时针旋转'
        angle=-angle
    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]   # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * math.fabs(math.sin(math.radians(angle))) + height * math.fabs(math.cos(math.radians(angle))))
    widthNew = int(height * math.fabs(math.sin(math.radians(angle))) + width * math.fabs(math.cos(math.radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1]>pt4[1]:
        pt2[1],pt4[1]=pt4[1],pt2[1]
    if pt1[0]>pt3[0]:
        pt1[0],pt3[0]=pt3[0],pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    imgOut = cv2.resize(imgOut, (image_height, image_width),)
    return imgOut, info

def smooth(image):
    '''将image图像中的垃圾使用矩阵框圈出，返回矩形框的四个顶点的坐标(box)'''
    ''''以图片左下角为原点,box[0]为左下角顶点,box[1]为左上角'''
    '''box[2]右下角,box[3]右上角,closed为边缘检测经过腐蚀膨胀后的图像'''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    _, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    
    return box, closed

def preprocess(img, mask):
    source = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.bitwise_and(img, mask)
    #削减图像中光照不均匀的影响
    img = unevenLightCompensate(img, blockSize=16)
    #边缘检测，得到二值化的边缘图像
    img = edge_demo(img)
    #获得能够圈出垃圾的矩形框的顶点坐标
    box, closed = smooth(img)
    if box is None and closed is None:
        return False, None, None
    center = ((box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4, (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4)
    img, _ = rotate(source, box)
    
    return True, source, img
