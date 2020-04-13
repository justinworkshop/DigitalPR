# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:58:37 2020
生成TF模型文件
@author: wei.zheng
"""

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import read_dataset

print('tensortflow version:{0}'.format(tf.__version__))
print('keras version:{0}'.format(tf.keras.__version__))

# mnist.mpz是28x28的手写数字图片和对应标签的数据集
dataPath = "mnist.mpz";
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(dataPath)


#可视化样本，下面是输出了训练集中前count个样本
read_dataset.showLabel(x_train, y_train, 0)
read_dataset.showImage(x_train, 10)


# =============================================================================
# 基础的前馈神经网络模型
# 把图片展平，这里指定了input_shape 参数，否则模型无法通过 model.save() 保存
# 全连接图层,，128个神经元，激活函数relu
#
# 输出层 ，10 个单元， 使用 Softmax 获得概率分布
# =============================================================================
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# =============================================================================
# 配置训练过程
# optimizer：训练过程的优化方法
# loss： 训练过程中使用的损失函数
# metrics： 训练过程中监测的指标
# =============================================================================
sgd = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5)

# 评估模型对样本数据的输出结果
val_loss, val_acc = model.evaluate(x_test, y_test)
print('模型的损失值:', val_loss)
print('模型的准确度:', val_acc)

# 保存成HDF5模型
keras_file = "keras_mnist_model.h5"
tf.keras.models.save_model(model, keras_file)

# 转成tflite模型
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("keras_mnist_model.tflite", "wb").write(tflite_model)