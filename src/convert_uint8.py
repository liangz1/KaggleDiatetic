import numpy as np


for i in range(1, 18):
    name="X_%d.npy" % (i*3000)
    X=np.load(name)
    np.save(name, X.astype('uint8'))
    name="Y_%d.npy" % (i*3000)
    X=np.load(name)
    np.save(name, X.astype('uint8'))