#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np

def plot_bbs(image, bbs, visualize=False):
    height, width, channels = image.shape
    cll, cx1, cy1, cwidth, cheight  = bbs
    #x->width y->height
    x1 = int(np.rint((cx1 - cwidth/2)*width))
    y1 = int(np.rint((cy1 - cheight/2)*height))
    x2 = int(np.rint((cx1 + cwidth/2)*width))
    y2 = int(np.rint((cy1 + cheight/2)*height))
    print (x1,y1)
    print (x2,y2)
    # Create a Rectangle patch
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)

    if visualize:
        cv2.imshow("TEST",image)
        cv2.waitKey()

if __name__ == "__main__":
    filepath =  "/media/datasets/simanimation/depth_testdataset_v3/files.txt"
    print(filepath)
    while os.path.exists(filepath):
        images = open(filepath,'r')
        for img_filename in images:
            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            fl = open(label_filename, "r")
            label = fl.readline().split(" ")
            fl.close()
            print (label)
            print (img_filename)
            img = cv2.imread(img_filename.rstrip())#, cv2.IMREAD_GRAYSCALE)
            print (img.shape)
            detections = [float(d) for d in label]
            plot_bbs(img, detections, visualize = True)
            print("NEXT")
        f.close()
