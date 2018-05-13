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

          #x=(7,1024,256)
          x = Reshape((x.shape[0], x.shape[1], x.shape[2]/2, x.shape[2]/2))(x)
          #x=(7,1024,128,128)
          x = Permute((1,3,2,4))(x)
          #x=(7,128,1024,128)
          x = Peshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))(x)
          #x=(7,1024*128,128)

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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[mean_sqrt_l2_error])
    return model

def mean_sqrt_l2_error(y_true, y_pred):
    res = K.mean(K.square((y_pred - y_true)**2), axis=-1)
    print 'mean_sqrt_l2_error', res,  y_true, y_pred
    return res

def signal_noise_rate(y_true, y_pred):
    Y = y_true
    P = y_pred
    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((P-Y)**2 + 1e-6, axis=[1,2]))
    sqrn_l2_norm = tf.sqrt(tf.reduce_mean(Y**2, axis=[1,2]))
    snr = 20 * tf.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.)
    avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
    avg_snr = tf.reduce_mean(snr, axis=0)
    return avg_snr

class MyDataGenerator(object):

    def create_generator(self, h5_path, batch_size):
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

def train(log_dir, model_dir, train_h5, val_h5, args):

    model = create_model()
    plot_model(model, to_file='model.png', show_shapes=True)

    gen = MyDataGenerator()
    train_gen = gen.create_generator(train_h5, args.batch_size)
    val_gen = gen.create_generator(val_h5, args.batch_size)

    md_cb = ModelCheckpoint(os.path.join(model_dir,'check.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1)
    tb_cb = MyTensorBoard(log_dir=log_dir)

    class PSNRCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs['loss'] * 255.
            val_loss = logs['val_loss'] * 255.
            psnr = 20 * math.log10(255. / math.sqrt(loss))
            val_psnr = 20 * math.log10(255. / math.sqrt(val_loss))
            print("\n")
            print("PSNR:%s" % psnr)
            print("PSNR(val):%s" % val_psnr)

    ps_cb = PSNRCallback()

    model.fit_generator(
        generator = train_gen,
        validation_data = val_gen,
        steps_per_epoch = args.steps,
        validation_steps = args.steps,
        epochs = args.epochs,
        callbacks=[md_cb, tb_cb, ps_cb])

    model.save(os.path.join(model_dir,'model.h5'))

class MyTensorBoard(TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        super(MyTensorBoard, self).on_epoch_end(epoch, logs)
        '''
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = 10
        summary_value.tag = 'snr'
        self.writer.add_summary(summary, epoch)
        '''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("model_dir")
    parser.add_argument("train_h5")
    parser.add_argument("val_h5")
    parser.add_argument('-e', '--epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('-s', '--steps', type=int, default=4, help='steps per epoch')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()
    print(args)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    train(args.log_dir, args.model_dir, args.train_h5, args.val_h5, args)
