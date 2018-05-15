from os.path import isfile, join, exists
from os import listdir, makedirs
import os
import numpy as np
import h5py
import math
import random
import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Input, Conv1D, Dropout, Add, Concatenate, UpSampling1D, Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
import keras.backend as K
from tensorflow.python import debug as tfdebug

# ----------------------------------------------------------------------------

sess = tf.Session()
sess = tfdebug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)


X = Input(shape=(6,4),name="my_input")
'''
(1,6,4)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
'''
x = X
s = (-1, int(x.shape[2])/4)
print s
#(-1, 1)

x = Permute((2,1), name='my_permute')(x)
print x.shape
#(?, 4, 6)
#output:
'''
[[0,4,8,12,16,20]
[1,5,9,13,17,21]
[2,6,10,14,18,22]
[3,7,11,15,19,23]]
'''

x = Reshape(s, name='my_reshape')(x)
print x.shape
#(?, 24, 1)
'''

'''

model = Model(inputs=X, outputs=x)
model.compile(optimizer='adam', loss='mean_squared_error')
plot_model(model, to_file='model.png', show_shapes=True)

train_x = np.arange(24).reshape((1,6,4))
train_y = np.arange(24).reshape((1,24,1))
print train_x
print train_y

model.fit(
        train_x,
        train_y,
        batch_size=1)

