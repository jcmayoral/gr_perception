#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import fileinput
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def read_docs(filepath):
    if os.path.exists(filepath):
        X = []
        y = []

        with open(filepath,'r') as files:
            for file in files:
                img_filename_r = file.rstrip()
                label_filename = file.replace(".jpg", ".txt").rstrip()
                if not os.path.exists(label_filename) or not os.path.exists(img_filename_r):
                    continue

                labels = open(label_filename, "r").readlines()
                for single_label in labels:
                    single_label_arr = single_label.split(" ")
                    f_label = [float(f) for f in single_label_arr]
                    f_label[0] = int(f_label[0])
                    y.append(f_label[0])
                    X.append(f_label[1:])

        X = np.asarray(X)
        y = np.asarray(y)
        return X,y

if __name__ == "__main__":
    rootpath = sys.argv[1]
    train_filepath = os.path.join(rootpath, "files_train.txt")
    valid_filepath = os.path.join(rootpath, "files_valid.txt")

    X_train, y_train = read_docs(train_filepath)
    X_valid, y_valid = read_docs(valid_filepath)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train,y_train)
    print "train score ", model.score(X_train, y_train)
    print "valid score ", model.score(X_valid, y_valid)

    cm = confusion_matrix(y_valid, model.predict(X_valid))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0,1,2,3), ticklabels=('Predicted Lethal', 'Danger', 'Warning', 'Safe'))
    ax.yaxis.set(ticks=(0,1,2,3), ticklabels=('Actual Lethal', 'Danger', 'Warning', 'Safe'))
    #ax.set_ylim(1.5, -0.5)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()

    print(classification_report(y_valid, model.predict(X_valid)))
