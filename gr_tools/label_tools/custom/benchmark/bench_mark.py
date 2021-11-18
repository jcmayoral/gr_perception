#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import fileinput
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.naive_bayes import GaussianNB

def parse_file_old(filepath):
    lines = open(filepath,'r').readlines()
    ddict = dict()
    for l in lines:
        key, value = l.split("=")
        ddict[key] = value.rstrip()
    return ddict

def parse_file(filepath):
    lines = open(filepath,'r').readlines()
    ddict = dict()
    for l in lines:
        if l[0] == "#" or len(l) == 1:
            continue
        key, value = l.strip().split(":")
        ddict[key] = value.strip()
    return ddict


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
                    print(single_label)
                    single_label_arr = single_label.split(" ")
                    f_label = [float(f) for f in single_label_arr]
                    f_label[0] = int(f_label[0])
                    y.append(f_label[0])
                    X.append(f_label[1:])

        X = np.asarray(X)
        y = np.asarray(y)
        return X,y

def special_metric(y_valid, y_pred):
    assert len(y_valid) == len(y_pred)
    counter = 0
    for a,b in zip(y_valid, y_pred):
        if np.fabs(a-b)>1:
            counter +=1.0
    print ("special metric to be Updated ", counter/len(y_valid))

if __name__ == "__main__":
    filepath = sys.argv[1]
    rootpath = "/".join(filepath.split("/")[:-2])
    #train_filepath = os.path.join(rootpath, "files_train.txt")
    #valid_filepath = os.path.join(rootpath, "files_valid.txt")
    ddict =  parse_file(filepath)


    train_filepath = os.path.join(rootpath, ddict["train"])
    valid_filepath = os.path.join(rootpath, ddict["val"])
    test_filepaht = os.path.join(rootpath, ddict["test"])

    X_train, y_train = read_docs(train_filepath)
    X_valid, y_valid = read_docs(valid_filepath)

    if sys.argv[2] == "logistics":
        model = LogisticRegression(solver='liblinear', random_state=0)

    if sys.argv[2] == "bayes":
        model = GaussianNB()
    model_name = sys.argv[2]
    #savename = sys.argv[3]

    model.fit(X_train,y_train)
    print ("train score ", model.score(X_train, y_train))
    print ("valid score ", model.score(X_valid, y_valid))
    y_pred = model.predict(X_valid)
    print ("mean absolute_error: ", mean_absolute_error(y_valid, y_pred))
    special_metric(y_valid, y_pred)

    sys.exit()
    cm = confusion_matrix(y_valid, model.predict(X_valid))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm, cmap = plt.cm.Blues)
    ax.grid(False)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")
    ax.xaxis.set(ticks=(0,1,2,3), ticklabels=('Lethal', 'Danger', 'Warning', 'Safe'))
    ax.yaxis.set(ticks=(0,1,2,3), ticklabels=('Lethal', 'Danger', 'Warning', 'Safe'))
    #ax.set_ylim(1.5, -0.5)
    thresh = 1500
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha='center', va='center', size=20, color="white" if  cm[i, j] > thresh else "black")
    plt.savefig("confussion_matrix_benchmark_{}.jpg".format(savename))
    plt.show()

    print(classification_report(y_valid, model.predict(X_valid)))
