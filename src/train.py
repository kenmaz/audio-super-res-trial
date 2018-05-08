'''
from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import SGD, Adam
import math
import pred
import os
import random
from PIL import Image
'''

from os.path import isfile, join, exists
from os import listdir, makedirs
import numpy as np
import h5py
import tensorflow as tf
from keras.utils import plot_model
from keras.layers.core import Activation
from keras.layers import Input, Conv1D, Dropout, Add, Concatenate, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

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
        images = []
        labels = []
        while True:
            files = listdir(input_dir)
            random.shuffle(files)
            for f in files:
                images.append(self.load_image(input_dir, f))
                labels.append(self.load_image(label_dir, f))
                if len(images) == batch_size:
                    x_inputs = np.asarray(images)
                    x_labels = np.asarray(labels)
                    images = []
                    labels = []
                    yield x_inputs, x_labels

    def load_image(self, src_dir, f):
        X = np.asarray(Image.open(join(src_dir, f)).convert('RGB'), dtype='float32')
        X /= 255.
        return X

def load_h5(h5_path):
  with h5py.File(h5_path, 'r') as hf:
    print 'List of arrays in input file:', h5_path, hf.keys()
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print 'Shape of X:', X.shape
  return X, Y

def train(log_dir, model_dir, train_h5, val_h5):

    x_train, y_train = load_h5(train_h5)
    x_val, y_val = load_h5(val_h5)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    resnet = create_model()
    plot_model(resnet, to_file='model.png', show_shapes=True)
    return

    datagen = MyDataGenerator()
    train_gen = datagen.flow_from_directory(os.path.join(
        train_dir, 'input'),
        os.path.join(train_dir, 'label'),
        batch_size = 10)

    val_gen = datagen.flow_from_directory(
        os.path.join(test_dir, 'input'),
        os.path.join(test_dir, 'label'),
        batch_size = 10)

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

    srcnn_model.fit_generator(
        generator = train_gen,
        steps_per_epoch = 10,
        validation_data = val_gen,
        validation_steps = 10,
        epochs = 1,
        callbacks=[pd_cb, ps_cb, md_cb, tb_cb])

    srcnn_model.save(os.path.join(model_dir,'model.h5'))

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
