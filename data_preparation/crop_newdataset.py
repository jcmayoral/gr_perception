#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

import fileinput
import time
import copy

DATABASE_PATH = "/home/jose/media/elsevier/devel_dataset"

def wh2xyxy(labels,h,w):
    new_labels = copy.copy(labels)
    cll, cx1, cy1, cwidth, cheight  = labels
    #x->width y->height
    new_labels[1] = np.max(int(np.rint((cx1 - cwidth/2)*w)),0)
    new_labels[2] = np.max(int(np.rint((cy1 - cheight/2)*h)),0)
    new_labels[3] = int(np.rint((cx1 + cwidth/2)*w))
    new_labels[4] = int(np.rint((cy1 + cheight/2)*h))

    new_labels[1:] = [int(d) for d in new_labels[1:]]
    #print (x1,y1)

    return new_labels

if __name__ == "__main__":
    filepath = sys.argv[1] #"/home/jose/datasets/real_iros2021/files.txt"
    masterfile = open(os.path.join(DATABASE_PATH, "master.txt"), "w")

    if os.path.exists(filepath):
        images = open(filepath,'r')

        start_index = 0

        for img_index, img_filename in tqdm(enumerate(images),desc="image"):
            img_filename = img_filename.rstrip()
            label_filename = img_filename.replace(".png", ".txt").rstrip()
            label_filename = label_filename.replace(".jpg", ".txt").rstrip()

            labels = []

            if not os.path.exists(label_filename):
                print " file {} does not exist".format(label_filename)
                continue

            if not os.path.exists(img_filename):
                print " file {} does not exist".format(img_filename)
                continue


            img = cv2.imread(img_filename)
            cv2.imshow("image", img)
            h,w,c = img.shape
            cv2.waitKey(50)

            labels = open(label_filename, "r").readlines()
            newtexts = []
            image_rootname = img_filename.split("/")[-1].split(".")[0]

            for d,i in enumerate(labels):
                label = [float(data) for data in i.strip().split(" ")]#)
                new_labels = wh2xyxy(label, h,w)
                cropped_image = img[new_labels[2]:new_labels[4], new_labels[1]:new_labels[3],:]

                cv2.imshow("cropped_image", cropped_image)
                new_imagefilename = os.path.join(DATABASE_PATH, image_rootname + "_" + str(d) +".jpg")
                new_labelfilename = os.path.join(DATABASE_PATH, image_rootname + "_" + str(d) +".txt")

                #this will create a single file by label
                labelfile = open(new_labelfilename, "w")
                labelfile.write(i)
                labelfile.close()


                cv2.imwrite(new_imagefilename, cropped_image)
                #append to master fill (all images paths)
                masterfile.write(new_imagefilename)
                cv2.waitKey(50)

    f.close()
