from matplotlib import pyplot as plt
import librosa
import numpy as np
import argparse
from scipy.signal import decimate
from scipy import interpolate
import tensorflow as tf
from keras.models import load_model
from keras.backend import tensorflow_backend
from keras.models import Model
from keras.layers import Input

def upsample_wav(wav, args, model):
  # load signal
  x_hr, fs = librosa.load(wav, sr=args.sr)

  num_layer = 5
  in_size = (len(x_hr)/2**num_layer)*(2**num_layer)
  print in_size
  x_hr = x_hr[0:in_size]
  print('original_wav',len(x_hr))

  # downscale signal
  x_lr = decimate(x_hr, args.r)
  print('donwsample_wav',len(x_lr))

  x_lr = upsample(x_lr, args.r)
  print('interpolate_wav',len(x_lr))

  P = model.predict(x_lr.reshape((1,len(x_lr),1)))
  x_pr = P.flatten()
  print('upsample_wav',len(x_pr))

  # crop so that it works with scaling ratio
  x_hr = x_hr[:len(x_pr)]
  x_lr = x_lr[:len(x_pr)]

  # save the file
  outname = wav + '.out'
  print outname

  save_wav(outname, x_hr, x_lr, x_pr, fs)

def save_wav(outname, x_hr, x_lr, x_pr, fs):
  librosa.output.write_wav(outname + '.hr.wav', x_hr, fs)
  librosa.output.write_wav(outname + '.lr.wav', x_lr, fs)
  librosa.output.write_wav(outname + '.pr.wav', x_pr, fs)

  # save the spectrum
  S = get_spectrum(x_pr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.pr.png')
  S = get_spectrum(x_hr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.hr.png')
  S = get_spectrum(x_lr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.lr.png')

def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  S = np.log1p(np.abs(S))
  return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
  plt.imshow(S.T, aspect=10)
  plt.tight_layout()
  plt.savefig(outfile)

def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  f = interpolate.splrep(i_lr, x_lr)
  x_sp = interpolate.splev(i_hr, f)
  return x_sp

def setup_session():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

def pred(model_h5, wav_file, args):
    setup_session()
    model = load_model(model_h5)
    model.summary()
    upsample_wav(wav_file, args, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_h5")
    parser.add_argument("wav_file")
    parser.add_argument('--r', help='upscaling factor', type=int, default=4)
    parser.add_argument('--sr', help='high-res sampling rate', type=int, default=16000)
    args = parser.parse_args()
    print(args)

    pred(args.model_h5, args.wav_file, args)
