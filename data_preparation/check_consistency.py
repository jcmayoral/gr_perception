#!/usr/bin/python3
from PIL import Image
import cv2
import sys
import os
import numpy as np
from tqdm import tqdm

files_path = sys.argv[1]

print "fie " , files_path
f = open(files_path)

with open("files_" + sys.argv[2] + "_filtered.txt", "a") as f2:
    for filepath in tqdm(f, desc="FILES "):
        filepath = filepath.rstrip()
        if not os.path.isfile(filepath):
            print ("FILE NOT FOUND {}".format(filepath))
            continue
        labelpath = filepath.replace("jpg", "txt")

        #first try npysave_rgba_image
        if not os.path.isfile(labelpath):
            print ("LABELS NOT FOUND {}".format(labelpath))
            continue
        f2.write(filepath + "\n")
f2.close()
f.close()
