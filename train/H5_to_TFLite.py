# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:35:58 2020

@author: qiqi
"""


import tensorflow as tf
from tensorflow.keras.models import load_model, Model

MobileNetV1 = load_model('models/MobileNetV1.h5')
MobileNetV1 = Model(inputs=MobileNetV1.input, outputs=MobileNetV1.get_layer('global_average_pooling2d').output)

converter = tf.lite.TFLiteConverter.from_keras_model(MobileNetV1)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open('models/MobileNetV1.tflite', 'wb').write(tflite_quant_model)