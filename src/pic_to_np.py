from PIL import Image
import numpy as np
import os
import json


def pic_to_np():
    data_dir = '/home/yunhan/kaggle_diabetic/test_tiny_224/'
    names = os.listdir(data_dir)
    print("%d files" % len(names))

    with open('label_dict.json') as f:
        label_dict = json.load(f)
    labels = []
    Xs = []
    batch_size = 3000
    for i, fname in enumerate(names):
        label = label_dict[fname[:-5]]
        labels.append(label)

        pic = Image.open(data_dir + fname)
        pix = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)
        Xs.append(pix)
        #print(pix)
        #print(label)
        #return  # debug
        if (i + 1) % batch_size == 0:
            np.save('X_%s.npy' % (i + 1), np.stack(Xs))
            np.save('Y_%s.npy' % (i + 1), np.array(labels))
            Xs = []
            labels = []


if __name__ == '__main__':
    pic_to_np()
