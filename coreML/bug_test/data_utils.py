import numpy as np
import cv2

def get_test_data_batches(data_dir='/Users/turbo_strong/Desktop/untitled_folder/'):
    for i in range(11):
        img_path = "%sX_%d.tiff" % (data_dir, i)
        image = cv2.imread(img_path)
        data = []
        if image is None:
            print("Failed to open {}".format(path))
            continue
        data.append(image)
        X = np.array(data, dtype=np.uint8)
        yield X / 255.

def get_test_data_batches_non_normalized(data_dir='/Users/turbo_strong/Desktop/untitled_folder/'):
    for i in range(11):
        img_path = "%sX_%d.tiff" % (data_dir, i)
        image = cv2.imread(img_path)
        data = []
        if image is None:
            print("Failed to open {}".format(path))
            continue
        data.append(image)
        X = np.array(data, dtype=np.uint8)
        yield X
