import os
import cv2
import copy
import numpy as np
import argparse

def increase_contrast(image, brightness=30, contrast=150):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    brightness  = 80 - np.mean(gray)
    img = np.int16(image)
    img = img * (contrast/127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def center_and_crop(image, size=512):
    ret = image.copy()
    x = ret.shape[0]
    y = ret.shape[1]
    l = min([x // 2, y // 2])
    crop = ret[ x // 2 - l : x // 2  + l,  y//2 - l : y//2 + l]
    crop = cv2.resize(crop, (size, size))
    return crop


def preprocess(image):
    ret = center_and_crop(image)
    ret = increase_contrast(ret)
    return ret

    
if __name__ == "__main__":
    # Sample Usage: python preprocess.py input_dir output_dir 0
    # mode 0: dump npz to output_dir quietly
    # mode 1: dump .jpegs to output_dir LOUDLY
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('mode', type=int)
    args = parser.parse_args()
    
    try:
        os.stat(args.output_dir)
    except:
        os.mkdir(args.output_dir) 
    
    data = []

    for root, dirs, files in os.walk(args.input_dir):
        for name in files:
            if ".jpeg" in name:
                path = root +"/" + name
                image = cv2.imread(path)
                res = preprocess(image)
                
                if res is None:
                    print("No fundus is detected in {}".format(path))
                    continue
                else:
                    if args.mode == 0:
                        data.append(res)
                    
                    if args.mode == 1:
                        print(args.output_dir + "/processed_" + name)
                        cv2.imwrite(args.output_dir + "/processed_" + name, res)
    
    data = np.array(data, dtype=np.uint8)
    # print(data.shape)
    np.savez(args.output_dir + "/input.npz")