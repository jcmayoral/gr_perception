#!/usr/bin/python3
from PIL import Image
import cv2
import sys
import os
import numpy as np

files_path = sys.argv[1]

f = open(files_path)

with open("files_" + sys.argv[2] + ".txt", "a") as f2:
    for line in f:
        filepath = line.rstrip()
        if not os.path.isfile(filepath):
            print ("FILE NOT FOUND {}".format(filepath))
            continue
        print ("FILE FOUND {}".format(filepath))
        depthpath = filepath.replace("image", "depthimage")
        depthpath = depthpath.replace("jpg", "npy")
        print(depthpath, "   AAA")
        #first try npy
        if not os.path.isfile(depthpath):
            print ("NOT FOUND ???")
            depthpath = filepath.replace("npy", "jpg")
            if not os.path.isfile(depthpath):
                #if not try jpg as worst case
                print ("DEPTH FILE NOT FOUND {}".format(depthpath))
                continue

        common_path = "/".join(filepath.split("/")[:-1])+"v2"
        print ("COMMON PATH", common_path)
        #Rename all files to png
        newfilename = filepath.split("/")[-1].split(".")[0] + ".png"
        print("filename " , newfilename)

        rgb_image = Image.open(filepath).convert('RGB')
        #rgb_image.show()

        original_shape = rgb_image.size
        print("s ", original_shape)
        print (depthpath)
        #b = Image.new('L', s, color=0)
        if "npy" in depthpath:
            print ("OK")
            depth_image = np.load(depthpath)
            depth_image = Image.fromarray(depth_image).convert('L')
            #sys.exit()
        else:
            depth_image = Image.open(depthpath)
        rgb_image.putalpha(depth_image)
        finalpath = os.path.join(common_path, newfilename)
        print ("final shape {} to {}".format(rgb_image.size, finalpath))
        rgb_image.save(finalpath)
        f2.write(finalpath + "\n")

        #i1 = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
        #i2 = cv2.imread("test.png",cv2.IMREAD_UNCHANGED)

        #cv2.imshow("4channels",i2)
        #cv2.imshow("3channels",i1)
        #cv2.imshow("depth?",i2[:,:,-1])
        #cv2.waitKey()
        #print (i1.shape,i2.shape)
        #output_image = Image.open("test.png")
        #output_image.show()
        #break
f2.close()
f.close()
