# -*- encoding:utf-8 -*-
from __future__ import division, print_function, absolute_import

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

"""
"""

$ python alexnet.py
---------------------------------
Run id: alexnet_oxflowers17
Log directory: tblogs_alexnet/
---------------------------------
Training samples: 1224
Validation samples: 136

### AdaGrad ### 収束遅いが悪くない
--
Training Step: 1800  | total loss: 0.89177
| AdaGrad | epoch: 090 | loss: 0.89177 - acc: 0.7170 | val_loss: 1.11424 - val_acc: 0.6176 -- iter: 1224/1224
--
Training Step: 1900  | total loss: 0.90449
| AdaGrad | epoch: 095 | loss: 0.90449 - acc: 0.7093 | val_loss: 1.09396 - val_acc: 0.6176 -- iter: 1224/1224
--
Training Step: 2000  | total loss: 0.87189
| AdaGrad | epoch: 100 | loss: 0.87189 - acc: 0.7230 | val_loss: 1.02559 - val_acc: 0.6324 -- iter: 1224/1224
--

### AdaDelta ###
機能している感じがしない

"""


import os, sys

import cctf
from cctf.layers.core import input_data, dropout, fully_connected
from cctf.layers.conv import conv_2d, max_pool_2d
from cctf.layers.normalization import local_response_normalization
from cctf.layers.estimator import regression

import cctf.datasets.oxflower17 as oxflower17
#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(224, 224))
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'
#network = input_data(shape=[None, 224, 224, 3])
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')

# 80 x 17 images = 1360 images/epoch
# 64 images/step
# 1360 / 64 = (images/epoch) / (images/step) = images*step / epoch*images = step / epoch
# 21.25 steps で 1 epoch
# 10 epochs で減衰させるのであれば、 1360*10/64

import math

num_images = 80 * 17 * 0.9 # 1224 validation set を除く
steps_per_epoch = int(math.ceil(num_images / 64)) # 19

""" constructor
// Momentum
def __init__(self, learning_rate=0.001,
                    momentum=0.9,
                    lr_decay=0.,
                    decay_step=100,
                    staircase=False,
                    use_locking=False,
                    name="Momentum"):
// AdaGrad
def __init__(self, learning_rate=0.001,
                    initial_accumulator_value=0.1,
                    use_locking=False, name="AdaGrad"):

// AdaDelta
def __init__(self, learning_rate=0.001,
                    rho=0.1,
                    epsilon=1e-08,
                    use_locking=False,
                    name="AdaDelta"):

// RMSProp
def __init__(self, learning_rate=0.001,
                    decay=0.9,
                    momentum=0.0,
                    epsilon=1e-10, use_locking=False, name="RMSProp"):
"""

#momentum = cctf.Momentum(learning_rate=0.005, momentum = 0.9, lr_decay=0.96,decay_step=steps_per_epoch*10)
#adagrad = cctf.AdaGrad(learning_rate=0.001)


#network = regression(network, optimizer=adagrad, loss='categorical_crossentropy')
#network = regression(network, optimizer=momentum, loss='categorical_crossentropy')

network = regression(network, optimizer='adagrad',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

#network = regression(network, optimizer='rmsprop',
#                     loss='categorical_crossentropy',
#                     learning_rate=0.001)

#network = regression(network, optimizer='momentum',
#                     loss='categorical_crossentropy',
#                     learning_rate=0.001)


model_path="model_alexnet_adagrad2"

os.mkdir(model_path)

# Training
model = cctf.DNN(network,
                    checkpoint_path="%s/model" % model_path,
                    max_checkpoints=1,
                    tensorboard_verbose=3,
                    tensorboard_dir="tblogs_alexnet_adagrad2")

# デフォルトは 1000 epochs
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=100,
          snapshot_epoch=False, run_id='alexnet_oxflowers17')
# model.fit(X, Y, n_epoch=5, validation_set=0.1, shuffle=True,
#          show_metric=True, batch_size=64, snapshot_step=200,
#          snapshot_epoch=False, run_id='alexnet_oxflowers17')

### EOF ###
