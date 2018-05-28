from os.path import isfile, join, exists
from os import listdir, makedirs
import os
import sys
import datetime
from datetime import timedelta, tzinfo
import numpy as np
import h5py
import math
import random
import argparse
import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Input, Conv1D, Dropout, Add, Concatenate, UpSampling1D, Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
import keras.backend as K
from tensorflow.python import debug as tfdebug

import audio_model

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        X = hf.get('data').value
        xs = X.shape
        X = X.reshape(-1,64,128,1)

        Y = hf.get('label').value
        ys = Y.shape
        Y = Y.reshape(-1,1,8192,1)
        return X, Y

class JST(tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=9)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return 'JST'

class S3SyncCallback(Callback):

    def __init__(self, log_dir):
        target = datetime.datetime.now(tz=JST()).strftime('%Y%m%d_%H%M%S')
        self.s3_path = 's3://tryswift/audio-super-resolution/%s' % target
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        self.sync()

    def sync(self):
        cmd = "aws s3 sync %s %s" % (self.log_dir, self.s3_path)
        print cmd
        res = (os.system(cmd) == 0)
        print res
        return res

def init_callbacks(model_dir, log_dir, s3_sync = False):
    callbacks = []
    md_cb = ModelCheckpoint(os.path.join(model_dir,'check.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1)
    callbacks.append(md_cb)

    tb_cb = TensorBoard(log_dir=log_dir)
    callbacks.append(tb_cb)

    if s3_sync:
        s3_cb = S3SyncCallback(log_dir = log_dir)
        if not s3_cb.sync():
            print 's3 sync failed'
            sys.exit()
        callbacks.append(s3_cb)

    print 'callbacks', callbacks
    return callbacks

def train(log_dir, model_dir, train_h5, val_h5, args):

    sess = tf.Session()
    #sess = tfdebug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    callbacks = init_callbacks(
            model_dir = model_dir,
            log_dir = log_dir,
            s3_sync = args.s3_sync)

    model = audio_model.create_model()
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    train_X, train_Y = load_h5(train_h5)
    val_X, val_Y = load_h5(val_h5)

    model.fit(
            x = train_X,
            y = train_Y,
            batch_size = args.batch_size,
            epochs = args.epochs,
            validation_data = (val_X, val_Y),
            callbacks=callbacks)

    model.save(os.path.join(model_dir,'model.h5'), include_optimizer = False )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("model_dir")
    parser.add_argument("train_h5")
    parser.add_argument("val_h5")
    parser.add_argument('-e', '--epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--s3_sync', action='store_true', help='s3 sync')
    args = parser.parse_args()
    print(args)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    train(args.log_dir, args.model_dir, args.train_h5, args.val_h5, args)
