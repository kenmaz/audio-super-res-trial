from os.path import isfile, join, exists
from os import listdir, makedirs
import os
import numpy as np
import h5py
import math
import random
import tensorflow as tf
from keras.utils import plot_model
from keras.layers.core import Activation
from keras.layers import Input, Conv1D, Dropout, Add, Concatenate, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

# ----------------------------------------------------------------------------

def create_model():
    X = Input(shape=(None,1))

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
          x = UpSampling1D(size=2)(x)
          x = Concatenate(axis=-1)([x, l_in])
          print 'U-Block-%d: %s' % (l, x.get_shape())

      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv1D(filters=1, kernel_size=9, padding='same', kernel_initializer='random_normal')(x)
        x = UpSampling1D(size=2)(x)
        print 'Last-Block-1: %s' % x.get_shape()

      x = Add()([x, X])

    model = Model(inputs=X, outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

class MyDataGenerator(object):

    def create_generator(self, h5_path, batch_size=32):
        while True:
            a, b = self.load_h5(h5_path)
            xy = zip(a,b)
            random.shuffle(xy)
            x_list = []
            y_list = []

            for (x, y) in xy:
                x_list.append(x.reshape(x.shape[1:]))
                y_list.append(y.reshape(y.shape[1:]))
                if len(x_list) == batch_size:
                    yield np.asarray(x_list), np.asarray(y_list)
                    x_list = []
                    y_list = []

            if len(x_list) > 0:
                yield np.asarray(x_list), np.asarray(y_list)

    def load_h5(self, h5_path):
      with h5py.File(h5_path, 'r') as hf:
        X = hf.get('data').value
        Y = hf.get('label').value
      return np.vsplit(X, X.shape[0]), np.vsplit(Y, Y.shape[0])

def train(log_dir, model_dir, train_h5, val_h5):

    model = create_model()
    plot_model(model, to_file='model.png', show_shapes=True)

    gen = MyDataGenerator()
    train_gen = gen.create_generator(train_h5)
    val_gen = gen.create_generator(val_h5)

    md_cb = ModelCheckpoint(os.path.join(model_dir,'check.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1)
    tb_cb = MyTensorBoard(log_dir=log_dir)

    model.fit_generator(
        generator = train_gen,
        validation_data = val_gen,
        steps_per_epoch = 10,
        validation_steps = 10,
        epochs = 3,
        callbacks=[md_cb, tb_cb])

    model.save(os.path.join(model_dir,'model.h5'))

class MyTensorBoard(TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
	for name, value in logs.items():
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		self.writer.add_summary(summary, epoch)

	summary = tf.Summary()
	summary_value = summary.value.add()
	summary_value.simple_value = 10
	summary_value.tag = 'snr'
	self.writer.add_summary(summary, epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("model_dir")
    parser.add_argument("train_h5")
    parser.add_argument("val_h5")
    args = parser.parse_args()
    print(args)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    train(args.log_dir, args.model_dir, args.train_h5, args.val_h5)
