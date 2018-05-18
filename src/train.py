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

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        X = hf.get('data').value
        Y = hf.get('label').value
        return X, Y

def train(log_dir, model_dir, train_h5, val_h5, args):

    sess = tf.Session()
    #sess = tfdebug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

    model = audio_model.create_model()
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    train_X, train_Y = load_h5(train_h5)
    val_X, val_Y = load_h5(val_h5)

    md_cb = ModelCheckpoint(os.path.join(model_dir,'check.h5'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1)
    tb_cb = TensorBoard(log_dir=log_dir)

    model.fit(
            x = train_X,
            y = train_Y,
            batch_size = args.batch_size,
            epochs = args.epochs,
            validation_data = (val_X, val_Y),
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
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()
    print(args)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    train(args.log_dir, args.model_dir, args.train_h5, args.val_h5, args)
