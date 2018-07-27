# -*- coding:utf-8 -*-
# Created by shellbye on 2018/7/27.
# https://tensorflow.google.cn/programmers_guide/low_level_intro?hl=zh-cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# 输入
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
# 输入对应的输出
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
# 网络\模型，一个对输入到输出关系的预测
linear_model = tf.layers.Dense(units=1)
# 模型的预测
y_pred = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

# print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
  _, loss_value = sess.run((train, loss))
  # print(loss_value)


print(sess.run(y_pred))