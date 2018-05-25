import sys
import numpy as np
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Reshape, Permute, Conv2D
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
    ]).reshape(-1, 1, 3, 1)
train_Y = np.array([
    [0,0,0],
    [1,0,0],
    [1,0,0],
    [2,0,0],
    [1,0,0],
    [2,0,0],
    [2,0,0],
    [3,0,0],
    ]).reshape(-1, 1, 3, 1)
print train_X.shape

def train():
    X = Input(shape=(1,None,1))
    x = X
    x = Conv2D(2, 2, padding='same')(x)
    x = Conv2D(1, 2, padding='same')(x)
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
    print train_Y.reshape(-1,3)
    print "*******"
    print res.reshape(-1,3)

model_h5_name = 'model.h5'
mlmodel_path = 'out/AudioSR.mlmodel'
mlmodel_input_name = 'wav'

def convert():
    coreml_model = coremltools.converters.keras.convert(model_h5_name, input_names = mlmodel_input_name)
    coreml_model.save(mlmodel_path)

def coreml():
    model = coremltools.models.MLModel(mlmodel_path)
    print model
    val = np.array([[0.0,1.0,0.0]]).reshape(1,3,1)
    print val
    print val.shape
    data = {mlmodel_input_name:val}
    res = model.predict(data)
    print res

eval(sys.argv[1])()

