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

    common_path = filepath.split("/")
    print ("COMMON PATH", common_path)

    rgb_image = Image.open(filepath).convert('RGB')
    rgb_image.show()

    original_shape = rgb_image.size
    print("s ", original_shape)
    #b = Image.new('L', s, color=0)
    depth_image = Image.open(depthpath)
    rgb_image.putalpha(depth_image)
    print ("s1", rgb_image.size)
    rgb_image.save('test.png')

    #i1 = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    #i2 = cv2.imread("test.png",cv2.IMREAD_UNCHANGED)

    #cv2.imshow("4channels",i2)
    #cv2.imshow("3channels",i1)
    #cv2.imshow("depth?",i2[:,:,-1])
    #cv2.waitKey()
    #print (i1.shape,i2.shape)
    output_image = Image.open("test.png")
    output_image.show()
    print (a.mode)
    break
f.close()
