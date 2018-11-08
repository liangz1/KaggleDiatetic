import os
import cv2
import numpy as np
import argparse


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def parse_frame(filename, form="mp4", frame_gap=1, verbose=True, threshold=300):
    vidcap = cv2.VideoCapture('.'.join([filename, form]))
    success,image = vidcap.read()
    count = 0
    success = True
    if os.path.exists("./" + filename) == False:
        os.mkdir("./"+filename)
    while success:
        if count % frame_gap == 0:
            if verbose:
                print(count)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output = image.copy()
            circles = cv2.HoughCircles(gray, cv2.cv2.HOUGH_GRADIENT, 1.3, 50, 
                                       param1=5,param2=110,minRadius=58,maxRadius=100)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                rad = max([r for (x, y, r) in circles])
                for (x, y, r) in circles:
                    if r == rad:
                        mask = np.zeros((output.shape[0],output.shape[1], output.shape[2]),dtype=np.uint8)
                        cv2.circle(mask,(x, y),r,(255,255,255),-1)
                        break
                output &= mask
                crop = output[y - r - 20:y + r + 20, x - r - 20:x + r + 20, :].copy()
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                hue = [crop[:, :, i].mean() for i in range(crop.shape[-1])]
                if gray is not None:
                    fm = variance_of_laplacian(gray)
                    if verbose:
                        print("Laplacian variance: " + str(fm))
                        print("Average color "+ str(hue))
                    if fm > threshold and hue[0] > 30 and hue[0] < 50 and hue[1] > 40 and hue[2] > 60:
                        cv2.imwrite("./" + filename + "/frame%d.jpg" % count, crop)
            
            else:
                if verbose:
                    print("No pupil is detected in fram {}\n".format(count))
        success, image = vidcap.read()
        count += 1
    return


##sample_usage: python preprocess.py Diabetic_Retinopathy_Screen False 300 1 mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("verbose", type=bool)
    parser.add_argument("threshold", type=int)
    parser.add_argument("frame_gap", type=int)
    parser.add_argument("form", type=str)
    args = parser.parse_args()
    parse_frame(args.filename,verbose=args.verbose, frame_gap=args.frame_gap, threshold=args.threshold)
