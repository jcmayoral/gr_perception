#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import fileinput
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confussion_matrix

if __name__ == "__main__":
    filepath = sys.argv[1] #"/home/jose/datasets/real_iros2021/files.txt"

    if os.path.exists(filepath):
        X = []
        y = []

        with open(filepath,'r') as new_file:
            img_filename_r = new_file.rstrip()
            label_filename = new_file.replace(".jpg", ".txt").rstrip()
            if not os.path.exists(label_filename) or not os.path.exists(img_filename_r):
                continue
            labels = open(label_filename, "r").readlines()

            for label in labels:
                f_label = [float(f) for f in label]
                f_label[0] = int(f_label[0])
                y.append(f_label[0])
                X.append(f_label[1:])

        X = np.asarray(X)
        y = np.asarray(y)
        print X.shape
