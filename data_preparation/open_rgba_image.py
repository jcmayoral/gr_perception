#!/usr/bin/python3
from PIL import Image
import cv2
import sys
import os

files_path = sys.argv[1]

f = open(files_path)
for line in f:
    filepath = line.rstrip()
    if not os.path.isfile(filepath):
        print ("FILE NOT FOUND {}".format(filepath))
        continue
    print ("FILE FOUND {}".format(filepath))
    depthpath = filepath.replace("image", "depthimage")
    print ("Depth FILE FOUND {}".format(depthpath))
    if not os.path.isfile(depthpath):
        print ("DEPTH FILE NOT FOUND {}".format(depthpath))
        continue

    common_path = "/".join(filepath.split("/")[:-1])+"v2"
    print ("COMMON PATH", common_path)
    #Rename all files to png
    newfilename = filepath.split("/")[-1].split(".")[0] + ".png"
    print("filename " , newfilename)

    #rgbd_image = Image.open(os.path.join(common_path, newfilename))
    #.convert('RGB')
    #rgbd_image.show()
    #i1 = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    i2 = cv2.imread(os.path.join(common_path, newfilename),cv2.IMREAD_UNCHANGED)

    #cv2.imshow("4channels",i2)
    #cv2.imshow("r",i2[:,:,0])
    #cv2.imshow("g",i2[:,:,1])
    cv2.imshow("rgb",i2[:,:,:2])

    cv2.imshow("depth?",i2[:,:,-1])
    cv2.waitKey()
    #print (i1.shape,i2.shape)
    #output_image = Image.open("test.png")
    #output_image.show()
    #break
f.close()
