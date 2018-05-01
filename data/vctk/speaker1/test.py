import h5py
import sys
import numpy as np
import librosa

def h5():
    h5_path = sys.argv[1]
    print h5_path

    with h5py.File(h5_path, 'r') as hf:
        print 'List of arrays in input file:', hf.keys()
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
        print 'Shape of X:', X.shape
        print 'Shape of Y:', Y.shape

def wav():
    path = sys.argv[1]
    print path
    x, fs = librosa.load(path, sr=16000)
    print x
    print len(x)
    print fs

wav()

