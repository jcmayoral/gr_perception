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

for match in f:
    rgb_index, depth_index = match.rstrip().split(" ")
    print (rgb_index, depth_index)
    filepath = rgb_template.format(rgb_index)
    if not os.path.isfile(filepath):
        print ("FILE NOT FOUND {}".format(filepath))
        continue
    print ("FILE FOUND {}".format(filepath))
    depthpath = depth_img_template.format(depth_index)
    depth_image = cv2.imread(depthpath,cv2.IMREAD_UNCHANGED)

    rgb_image = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    print (rgb_image.shape)
    depth_3chanels = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

    rgbd = np.hstack((rgb_image,depth_3chanels))
    print (rgb_image.dtype, depth_image.dtype)
    cv2.imshow("rgbd", rgbd)

    #cv2.imshow("depth?",i2[:,:,-1])
    cv2.waitKey(20)


f.close()
