import argparse
import os.path
import sys
import os
import coremltools
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

model = coremltools.models.MLModel(args.model)
print model

val = np.array([[0.0,1.0,0.0]]).reshape(1,3,1)
print val, val.shape

data = {'wav':val}
res = model.predict(data)
print res

