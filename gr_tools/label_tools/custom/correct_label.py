#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

import fileinput
import time

def replace_line(file_name, line_nums, texts):
    lines = open(file_name, 'r').readlines()
    out = open(file_name, 'w')
    for line_nums, texts in zip(line_nums, texts):
        lines[line_nums] = texts[0]
    out.writelines(lines)
    out.close()

def replace_file(file_name, texts):
    out = open(file_name, 'w')
    for t in texts:
        out.write(t)
    out.close()


if __name__ == "__main__":
    filepath =  "/home/jose/datasets/real_iros2021/files.txt"

    if os.path.exists(filepath):
        images = open(filepath,'r')

        start_index = 0

        for img_index, img_filename in tqdm(enumerate(images)):
            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            labels = []

            if not os.path.exists(label_filename):
                continue

            lindx = 0
            replace_ind = []
            replace_data = []

            labels = open(label_filename, "r").readlines()
            print label_filename
            print labels
            newtexts = []

            for d,i in enumerate(labels):
                label = [data for data in i.strip().split(" ")]#)
                label[0] = int(float(label[0]))
                newlabel = ""
                newlabel = newlabel.join([str(c)+ " " for c in label])[:-1]+"\n"
                newtexts.append(newlabel)

            replace_file(label_filename, newtexts)
