import numpy as np
import tensorflow as tf

from scipy import interpolate
from model import Model, default_opt

from layers.subpixel import SubPixel1D, SubPixel1D_v2

from keras import backend as K
from keras.layers.core import Activation
from keras.layers import Conv1D, Dropout, Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# ----------------------------------------------------------------------------

class AudioUNet(Model):
  """Generic tensorflow model training code"""

  def __init__(self, from_ckpt=False, n_dim=None, r=2,
               opt_params=default_opt, log_prefix='./run'):
    # perform the usual initialization
    self.r = r
    Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                   opt_params=opt_params, log_prefix=log_prefix)

  def create_model(self, n_dim, r):
    # load inputs
    X, _, _ = self.inputs
    K.set_session(self.sess)

    with tf.name_scope('generator'):
      x = X
      L = self.layers
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print 'building model...'

      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
          conv1d = Conv1D(filters=nf, kernel_size=fs, padding='same', kernel_initializer='orthogonal', strides=2)
          x = conv1d(x)
          x = LeakyReLU(0.2)(x)
          print 'D-Block: ', x.get_shape()
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
          conv1d = Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], padding='same', kernel_initializer='orthogonal', strides=2)
          x = conv1d(x)
          x = Dropout(rate=0.5)(x)
          x = LeakyReLU(0.2)(x)

      # upsampling layers
      for l, nf, fs, l_in in reversed(zip(range(L), n_filters, n_filtersizes, downsampling_l)):
        with tf.name_scope('upsc_conv%d' % l):
          conv1d = Conv1D(filters=2*nf, kernel_size=fs, padding='same', kernel_initializer='orthogonal')
          x = (conv1d)(x)
          x = Dropout(rate=0.5)(x)
          x = Activation('relu')(x)
          x = SubPixel1D(x, r=2)
          x = Concatenate()([x, l_in])
          print 'U-Block: ', x.get_shape()

      # final conv layer
      with tf.name_scope('lastconv'):
        x = Conv1D(filters=2, kernel_size=9, padding='same', kernel_initializer='random_normal')(x)
        x = SubPixel1D(x, r=2) 
        print x.get_shape()

      g = Add()([x, X])

    return g

  def predict(self, X):
    assert len(X) == 1
    print 1,X.shape
    x_sp = spline_up(X, self.r)
    print 2,x_sp.shape
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
    print 3,x_sp.shape
    X = x_sp.reshape((1,len(x_sp),1))
    print 4,x_sp.shape
    feed_dict = self.load_batch((X,X), train=False)
    print 'predict', X.shape
    return self.sess.run(self.predictions, feed_dict=feed_dict)

# ----------------------------------------------------------------------------
# helpers

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp
