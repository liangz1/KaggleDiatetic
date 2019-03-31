import numpy as np

y_all_true=[]
for i in range(1, 18):
    name="Y_%d.npy" % (i*3000)
    Y=np.load(name)
    y_all_true.append(Y)
np.save('Y_51000_true.npy', np.vstack(y_all_true))