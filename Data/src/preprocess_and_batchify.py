import os
import cv2
import copy
import numpy as np
import pandas as pd
import argparse


def increase_contrast(image, contrast=150):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    brightness  = 100 - np.mean(gray)
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
    crop = np.array(crop, dtype=np.float32)
    crop = cv2.resize(crop, (size, size))
    crop = np.array(crop, dtype=np.uint8)
    return crop


def preprocess(image):
    ret = center_and_crop(image)
    ret = increase_contrast(ret)
    return ret


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('mode', type=int)
    args = parser.parse_args()
    
    try:
        os.stat(args.output_dir)
    except:
        os.mkdir(args.output_dir) 
    
    data = []
    label = []
    batch_idx = 0
    
    df = pd.read_csv("retinopathy_solution.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    for idx in range(len(df)):
        entry = df.iloc[[idx]]
        path = args.input_dir + entry['image'] + ".jpeg"
        y = entry['level']
        image = cv2.imread(path)
        res = preprocess(image)
        
        if idx % args.batch_size == 0 and idx != 0:
            data = np.array(data, dtype=np.uint8)
            batch_idx += 1
            print("Saving batch {} out of {}".format(batch_idx, 54))
            np.save("{}/X{}.npy".format(args.output_dir, batch_idx), data)
            np.save("{}/y{}.npy".format(args.output_dir, batch_idx), label)
            data = []
            label = []
        
        if args.mode == 0:
            data.append(res)
            label.append(y)
                
        if args.mode == 1:
            cv2.imwrite(args.output_dir + "/" + entry['image'] + ".jpeg", res)
    
    data = np.array(data, dtype=np.uint8)
    batch_idx += 1
    print("Saving batch {} out of {}".format(batch_idx, 54))
    np.save("{}/X_{}.npy".format(args.output_dir, batch_idx), data)
    np.save("{}/y{}.npy".format(args.output_dir, batch_idx), label)