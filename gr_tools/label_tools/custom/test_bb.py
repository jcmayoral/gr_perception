#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

def plot_bbs(image, bbs, visualize=False, out=None):
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
    cv2.putText(image, "RING_" +str(int(cll)), (int(cx1*width),int(cy1*height)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)

    if visualize:
        cv2.imshow("TEST",image)
        cv2.waitKey(25)

    if out is not None:
        out.write(image)

if __name__ == "__main__":
    filepath =  "/home/jose/datasets/real_iros2021/files.txt"
    out = cv2.VideoWriter('dataset.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
    counter = [0,0,0,0]

    if os.path.exists(filepath):
        images = open(filepath,'r')
        for img_filename in tqdm(images):
            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            labels = []
            if not os.path.exists(label_filename):
                print "label file {} not exists".format(label_filename)
                continue

            with open(label_filename, "r") as fl:
                labels = [data.strip().split(" ") for data in fl]#)
            fl.close()
            #print (label)
            img = cv2.imread(img_filename.rstrip().replace("txt", "jpg"))#, cv2.IMREAD_GRAYSCALE)

            for l in labels:
                detections = [float(d) for d in l]
                plot_bbs(img, detections, visualize = False, out=out)
                cl_ = int(detections[0])
                if cl_ < 0 or  cl_ > 3:
                    print "ERRROR.....",cl_, img_filename
                    continue
                counter[cl_] +=1
            #print("NEXT")
        #f.close()
    out.release()
    print "FINAL COUNTER", counter