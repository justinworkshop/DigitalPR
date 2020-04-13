# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:58:37 2020
生成TF模型文件
@author: wei.zheng
"""

import tensorflow.compat.v1 as tf

print('tensortflow version:{0}'.format(tf.__version__))
print('keras version:{0}'.format(tf.keras.__version__))

# mnist.mpz是28x28的手写数字图片和对应标签的数据集
dataPath = "mnist.mpz";
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(dataPath)

# 数据正规化
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


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

model.compile(optimizer='adam',  # 默认的较好的优化器
              loss='sparse_categorical_crossentropy',  # 评估“错误”的损失函数，模型应该尽量降低损失
              metrics=['accuracy'])  # 评价指标

# 训练模型
model.fit(x_train, y_train, epochs=3)

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