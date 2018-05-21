import sys
import numpy as np
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Reshape
from keras.models import Model
from keras.models import load_model
import coremltools

train_X = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1],
    ]).reshape(-1,3,1)
train_Y = np.array([
    [0,0,0],
    [1,0,0],
    [1,0,0],
    [2,0,0],
    [1,0,0],
    [2,0,0],
    [2,0,0],
    [3,0,0],
    ]).reshape(-1,3,1)
print train_X.shape

def train():
    X = Input(shape=(None,1))
    x = X
    x = Conv1D(3,3,padding='same',activation='relu')(x)
    x = Conv1D(1,3,padding='same',activation='relu')(x)
    model = Model(inputs=X, outputs=x)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_X, train_Y, epochs=1000)
    model.save('model.h5', include_optimizer = False )
    model.summary()

def pred():
    model = load_model('model.h5')
    model.summary()
    res = model.predict(train_X)
    print train_Y
    print "*******"
    print res

def convert():
    coreml_model = coremltools.converters.keras.convert('model.h5', input_names = 'wav')
    coreml_model.save('out/AudioSR.mlmodel')

def coreml():
    model = coremltools.models.MLModel('out/AudioSR.mlmodel')
    val = np.array([[1.0,1.0,1.0]]).reshape(-1,3,1)
    print val.shape
    data = {'wav':val}
    res = model.predict(data)
    print res

eval(sys.argv[1])()

