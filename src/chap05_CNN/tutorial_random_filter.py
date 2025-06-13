#!/usr/bin/env python
# coding: utf-8
# ## 随机 filter

# In[191]:
# 导入必要的库
import os # 提供操作系统相关功能，如文件路径操作
import tensorflow as tf
from tensorflow import keras # 引入Keras高级API接口
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten # 引入全连接层、Dropout层和展平层
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pylab # Matplotlib的绘图接口
from PIL import Image
import numpy as np # 数值计算库

# 定义一个简单的卷积模型
class MyConvModel(keras.Model): # 定义一个继承自Keras模型基类的自定义卷积神经网络模型
    def __init__(self):
        super(MyConvModel, self).__init__() # 调用父类的构造函数
        self.l1_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same')
# 使用 TensorFlow 的 tf.function 装饰器，将函数编译为 TensorFlow 图执行，提高性能   
    @tf.function
    def call(self, x):
        h1 = self.l1_conv(x)
        return h1# 直接返回卷积结果


# In[192]:
random_conv = MyConvModel()# 实例化一个新的卷积神经网络模型

# 打开一张尺寸为 639x516 的随机图片
# 使用Pillow库以二进制模式打开图片文件
img = Image.open(open('corgi.jpg', 'rb'))  # 返回PIL.Image对象

# 将PIL图像转换为numpy数组，并指定数据类型为float64
img = np.asarray(img, dtype='float32') / 255.0  # 归一化到0-1范围，使用float32更高效

# 在数组的第0维添加一个维度（批处理维度）
# 将形状从[H,W,C]变为[1,H,W,C]，符合模型输入要求
img = np.expand_dims(img, axis=0)

# 对图像应用随机卷积变换
# random_conv是自定义的卷积操作函数，可能用于数据增强
img_out = random_conv(img)  # 输出形状保持[1,H,W,C]

#使用pylab（通常是matplotlib.pyplot的别名）来创建一个包含四个子图的图形，并显示图像数据。
pylab.figure(figsize=(10, 7))# 创建一个绘图窗口，并设置图像显示的画布大小为 10x7 英寸
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img[0, :, :, :])        # 第一个子图：绘制原始输入图像
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 0])    # 第二个子图：显示输出图像的第一个通道（feature map）
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 1])    # 第三个子图：显示输出图像的第二个通道（feature map）
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 2])    # 第四个子图：显示输出图像的第三个通道（feature map）
pylab.show()                                                                    # 显示所有绘制的图像
