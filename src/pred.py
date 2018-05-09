'''
from os.path import isfile, join, exists
from os import listdir, makedirs
import os
import h5py
import math
import random
import tensorflow as tf
from keras.utils import plot_model
from keras.layers.core import Activation
from keras.layers import Input, Conv1D, Dropout, Add, Concatenate, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
'''

import numpy as np
import argparse
from keras.models import load_model
from keras.backend import tensorflow_backend
import tensorflow as tf
from models.io import upsample_wav

from keras.models import Model
from keras.layers import Input
# ----------------------------------------------------------------------------

def create_model():
    X = Input(shape=(8192,1))

    with tf.name_scope('generator'):
      x = X
      L = 4
      n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print 'building model...'
      print 'Input: %s' % x.get_shape()

      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
          x = Conv1D(filters=nf, kernel_size=fs, padding='same', kernel_initializer='orthogonal', strides=2)(x)
          x = LeakyReLU(0.2)(x)
          print 'D-Block-%d: %s' % (l, x.get_shape())
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
          x = Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], padding='same', kernel_initializer='orthogonal', strides=2)(x)
          x = Dropout(rate=0.5)(x)
          x = LeakyReLU(0.2)(x)
          print 'B-Block: ', x.get_shape()

      # upsampling layers
      for l, nf, fs, l_in in reversed(zip(range(L), n_filters, n_filtersizes, downsampling_l)):
        with tf.name_scope('upsc_conv%d' % l):
          x = Conv1D(filters=nf, kernel_size=fs, padding='same', kernel_initializer='orthogonal')(x)
          x = Dropout(rate=0.5)(x)
          x = Activation('relu')(x)
          l_in = Conv1D(filters=nf, kernel_size=fs, padding='same', kernel_initializer='orthogonal', strides=2)(l_in)
          #x = UpSampling1D(size=2)(x)
          x = Concatenate(axis=1)([x, l_in])
          print 'U-Block-%d: %s' % (l, x.get_shape())

      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv1D(filters=1, kernel_size=9, padding='same', kernel_initializer='random_normal')(x)
        x = UpSampling1D(size=2)(x)
        x = Activation('relu')(x)
        print 'Last-Block-1: %s' % x.get_shape()

      x = Add()([x, X])

    model = Model(inputs=X, outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

class MyDataGenerator(object):

    def flow_from_directory(self, h5_path, batch_size=32):
        images, labels = load_h5(h5_path)
        items = zip(images, labels)
        images = []
        labels = []
        while True:
            random.shuffle(items)
            for x, y in items:
                yield x, y

def load_h5(h5_path):
  with h5py.File(h5_path, 'r') as hf:
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print 'load_h5', X.shape, Y.shape
  return X, Y

def train(log_dir, model_dir, train_h5, val_h5):

    model = create_model()
    plot_model(model, to_file='model.png', show_shapes=True)

    train_X, train_Y = load_h5(train_h5)
    val_X, val_Y = load_h5(val_h5)

    class PredCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            pass
            #pred.predict(self.model, eval_img, 'base-%d.png' % epoch, 'out-%d.png' % epoch, False)

    class PSNRCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs['loss'] * 255.
            val_loss = logs['val_loss'] * 255.
            psnr = 20 * math.log10(255. / math.sqrt(loss))
            val_psnr = 20 * math.log10(255. / math.sqrt(val_loss))
            print("\n")
            print("PSNR:%s" % psnr)
            print("PSNR(val):%s" % val_psnr)

    pd_cb = PredCallback()
    ps_cb = PSNRCallback()
    md_cb = ModelCheckpoint(os.path.join(model_dir,'check.h5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
    tb_cb = TensorBoard(log_dir=log_dir)

    model.fit(
            train_X,
            train_Y,
            validation_data=(val_X, val_Y),
            steps_per_epoch = 10,
            validation_steps = 10,
            epochs = 1,
            callbacks=[pd_cb, ps_cb, md_cb, tb_cb])

    model.save(os.path.join(model_dir,'model.h5'))

def load_model_dummy(path):
    X = Input(shape=(8192,1))
    model = Model(inputs=X, outputs=X)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def setup_session():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

def pred(model_h5, wav_file, args):
    setup_session()
    model = load_model(model_h5)
    upsample_wav(wav_file, args, model)
    #res = model.predict(wav)
    #print(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_h5")
    parser.add_argument("wav_file")
    parser.add_argument('--r', help='upscaling factor', type=int, default=4)
    parser.add_argument('--sr', help='high-res sampling rate', type=int, default=16000)
    args = parser.parse_args()
    print(args)

    pred(args.model_h5, args.wav_file, args)
