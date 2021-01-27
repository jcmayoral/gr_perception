#!/usr/bin/python3
import cv2
import os
import sys

person_id = sys.argv[1]
img_index = int(sys.argv[2])
print(img_index, person_id)
image_id = "image_" + str(img_index)
filespath = os.path.join("testdataset", person_id,"Forward",image_id)
print(filespath)
while os.path.exists(filespath+".txt"):
    f = open(filespath+".txt", "r")
    detections = f.readline().split(" ")
    img = cv2.imread(filespath+'.jpg')#, cv2.IMREAD_GRAYSCALE)
    print (detections)
    print (img.shape)
    height, width, channels = img.shape
    detections = [float(d) for d in detections]
    print (detections)
    cll, cx1, cy1, cwidth, cheight  = detections
    #x->width y->height
    x1 = int((cx1 - cwidth/2)*width)
    x2 = int((cx1 + cwidth/2)*width)
    y1 = int((cy1 - cheight/2)*height)
    y2 = int((cy1 + cheight/2)*height)
    print (x1,y1)
    print (x2,y2)
    # Create a Rectangle patch
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.imshow("TEST",img)
    cv2.waitKey()
    img_index = img_index + 1
    image_id = "image_" + str(img_index)
    filespath = os.path.join("testdataset", person_id,"Forward",image_id)
    print(filespath)
