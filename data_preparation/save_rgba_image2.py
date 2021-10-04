#!/usr/bin/python3
from PIL import Image
import cv2
import sys
import os
import numpy as np


#INPUT IS A FILE WITH MATCHING INDECES
files_path = sys.argv[1]

images_path = "/".join(files_path.split("/")[:-1])
depth_img_template = os.path.join(images_path, "depthimage_{}.jpg")
rgb_template = os.path.join(images_path, "image_{}.jpg")


f = open(files_path)

with open("files_" + sys.argv[2] + ".txt", "a") as f2:
    for match in f:
        rgb_index, depth_index = match.rstrip().split(" ")
        print (rgb_index, depth_index)
        filepath = rgb_template.format(rgb_index)
        if not os.path.isfile(filepath):
            print ("FILE NOT FOUND {}".format(filepath))
            continue
        print ("FILE FOUND {}".format(filepath))
        depthpath = depth_img_template.format(depth_index)
        depthpath = depthpath.replace("jpg", "npy")
        print(depthpath, "   AAA")

        #first try npysave_rgba_image
        if not os.path.isfile(depthpath):
            print ("NOT FOUND ???" , depthpath)
            depthpath = depthpath.replace("npy", "jpg")
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
            print ("Numpy file")
            depth_image = np.load(depthpath)
            print ("DATA SHAPE ", depth_image.shape)
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
