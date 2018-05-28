from os.path import isfile, join, exists
from os import listdir, makedirs
import os
import numpy as np
import h5py
import math
import random
import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Input, Conv1D, Conv2D, Dropout, Add, Concatenate, UpSampling1D, Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import Adam
import keras.backend as K
from tensorflow.python import debug as tfdebug

# ----------------------------------------------------------------------------

def create_model():
    X = Input(shape=(64,128,1))

    with tf.name_scope('generator'):
      X2 = Reshape((1,-1,1))(X)

      x = X2
      L = 4
      n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print 'building model...'
      print 'Input: %s' % x.get_shape()

      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
          x = Conv2D(filters=nf, kernel_size=(1,fs), padding='same', kernel_initializer='orthogonal', strides=2)(x)
          x = LeakyReLU(0.2)(x)
          print 'D-Block-%d: %s' % (l, x.get_shape())
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
          x = Conv2D(filters=n_filters[-1], kernel_size=(1,n_filtersizes[-1]), padding='same', kernel_initializer='orthogonal', strides=2)(x)
          x = Dropout(rate=0.5)(x)
          x = LeakyReLU(0.2)(x)
          print 'B-Block: ', x.get_shape()

      # upsampling layers
      for l, nf, fs, l_in in reversed(zip(range(L), n_filters, n_filtersizes, downsampling_l)):
        with tf.name_scope('upsc_conv%d' % l):
          x = Conv2D(filters=nf*2, kernel_size=(1,fs), padding='same', kernel_initializer='orthogonal')(x)
          x = Dropout(rate=0.5)(x)
          x = Activation('relu')(x)
          x = subpixel1D(x, r=2)

          # CoreML not support 3d concatenate(axis=-1), so following error was happend.
          # >> raise ValueError('Only channel and sequence concatenation are supported.')
          # workaround: temporary reshape to 4-d, before concat back to 3-d

          #s = (1,-1,int(x.shape[-1]))
          #x = Reshape(s)(x)
          #l_in = Reshape(s)(l_in)
          #x = Concatenate(axis=-1)([x, l_in])
          #s = (-1,int(x.shape[-1]))
          #x = Reshape(s)(x)

          print 'U-Block-%d: %s' % (l, x.get_shape())

      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv2D(filters=2, kernel_size=(1,9), padding='same', kernel_initializer='random_normal')(x)
        x = subpixel1D(x, r=2)
        print 'Last-Block-1: %s' % x.get_shape()

      x = Add()([x, X2])

    model = Model(inputs=X, outputs=x)
    adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss=mean_sqrt_l2_error, metrics=[mean_sqrt_l2_error, signal_noise_rate])
    return model

def subpixel1D(x, r=2):
    print ('subpixel1D',x.shape)
    shape = (1, -1, int(x.shape[-1]/r))
    y = Reshape(shape)(x)
    print ('subpixel1D',y.shape)
    return y

def mean_sqrt_l2_error(y_true, y_pred):
    loss, snr = _calc_metrics(y_true, y_pred)
    return loss

def signal_noise_rate(y_true, y_pred):
    loss, snr = _calc_metrics(y_true, y_pred)
    return snr

def _calc_metrics(y_true, y_pred):
    Y = y_true
    P = y_pred
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((P-Y)**2 + 1e-6, axis=[1,2]))
    sqrn_l2_norm = tf.sqrt(tf.reduce_mean(Y**2, axis=[1,2]))
    snr = 20 * tf.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.)
    avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
    avg_snr = tf.reduce_mean(snr, axis=0)
    return avg_sqrt_l2_loss, avg_snr

