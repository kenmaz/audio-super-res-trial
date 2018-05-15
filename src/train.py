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

import audio_model

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

    sess = tf.Session()
    #sess = tfdebug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    model = audio_model.create_model()
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
    tb_cb = TensorBoard(log_dir=log_dir)

    model.fit_generator(
        generator = train_gen,
        validation_data = val_gen,
        steps_per_epoch = args.steps,
        validation_steps = args.steps,
        epochs = args.epochs,
        callbacks=[md_cb, tb_cb])

    model.save(os.path.join(model_dir,'model.h5'), include_optimizer = False )

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
