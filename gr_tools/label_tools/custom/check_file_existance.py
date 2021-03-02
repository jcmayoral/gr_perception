#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

import fileinput
import time

if __name__ == "__main__":
    filepath = sys.argv[1] #"/home/jose/datasets/real_iros2021/files.txt"

    if os.path.exists(filepath):
        images = open(filepath,'r')
        start_index = 0

        with open(filepath+".filter",'a') as new_file:
            for img_index, img_filename in tqdm(enumerate(images)):
                label_filename = img_filename.replace(".jpg", ".txt").rstrip()
                if not os.path.exists(label_filename):
                    continue
                #print label_filename
                new_file.write(label_filename+"\n")
