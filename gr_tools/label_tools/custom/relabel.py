#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

def plot_bbs(image, bbs):
    height, width, channels = image.shape
    cll, cx1, cy1, cwidth, cheight  = bbs
    #x->width y->height
    x1 = int(np.rint((cx1 - cwidth/2)*width))
    y1 = int(np.rint((cy1 - cheight/2)*height))
    x2 = int(np.rint((cx1 + cwidth/2)*width))
    y2 = int(np.rint((cy1 + cheight/2)*height))
    #print (x1,y1)
    #print (x2,y2)
    #print (cll, type(cll))
    # Create a Rectangle patch
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image, str(int(cll)), (int(cx1*width),int(cy1*height)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)

    cv2.imshow("TEST",image)
    key = cv2.waitKey(0)
    #if key == "32":
    #print "OK LLLLLL " + str(key)
    if key != 27 and key !=141:
        cv2.putText(image, "WAIT FOR NEW CLASS" +str(int(cll)), (int(cx1*width),int(cy1*height)), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255),2)
        cv2.imshow("TEST",image)
        cv2.waitKey(1000)
    cv2.waitKey(1000)

    return key


import fileinput
import time

def replace_line(file_name, line_nums, texts):
    lines = open(file_name, 'r').readlines()
    out = open(file_name, 'w')
    for line_nums, texts in zip(line_nums, texts):
        lines[line_nums] = texts
    out.writelines(lines)
    out.close()

if __name__ == "__main__":
    filepath =  "/home/jose/datasets/dummy/dummy"

    print "WAIT 3 seconds before start"
    time.sleep(3)
    print "OK"

    if os.path.exists(filepath):
        images = open(filepath,'r')

        start_index = 0

        if os.path.exists("lastindex.txt"):
            read_mode = "r"
            start_index = int(open("lastindex.txt",read_mode).readline().rstrip())
            print "start index"


        lastindex_file = open("lastindex.txt","w")
        skipped = open("skipped.txt","a")

        for img_index, img_filename in tqdm(enumerate(images)):
            if img_index <= start_index:
                continue

            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            labels = []
            if not os.path.exists(label_filename):
                print "label file {} not exists".format(label_filename)
                continue

            with open(label_filename, "r") as fl:
            #for line in fileinput.input(label_filename, inplace=True):
                #print line
                replace_ind = []
                replace_data = []

                for index, line in enumerate(fl):
                    img = cv2.imread(img_filename.rstrip())
                    label = [data for data in line.strip().split(" ")]#)
                    print " FILE NAME ", label_filename
                    detections = [float(d) for d in label]
                    key = plot_bbs(img, detections)

                    if key == 141:
                        continue
                    if key == 27:
                        lastindex_file.close()
                        sys.exit()

                    label[0] = key - 176
                    newlabel = ""
                    newlabel = newlabel.join([str(c)+ " " for c in label])[:-1]

                    replace_ind.append(index)
                    replace_data.append(newlabel)

                #print "Image index", img_index
                lastindex_file.seek(0)
                lastindex_file.write(str(img_index))
                #print "indeces", replace_ind
                #print "data ", replace_data
                replace_line(label_filename,replace_ind,replace_data)

        lastindex_file.close()
                #print newlabel.rstrip()
