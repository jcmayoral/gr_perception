#!/usr/bin/python3
#import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tqdm import tqdm

import logging

def get_logger(    
        LOG_FORMAT     = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        LOG_NAME       = '',
        LOG_FILE_INFO  = 'file.log',
        LOG_FILE_ERROR = 'file.err'):

    log           = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)

    file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='w')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_error = logging.FileHandler(LOG_FILE_ERROR, mode='w')
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    log.setLevel(logging.INFO)

    return log

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


def read_docs(filepath, padding):
    if os.path.exists(filepath):
        X = []
        y = []
        types = [0,0,0,0]
        pad = -16

        with open(filepath,'r') as files:
            for file in tqdm(files):
                pad += 1
                if pad%padding != 0:
                    continue
                img_filename_r = file.rstrip()
                label_filename = file.replace(".jpg", ".txt").rstrip()
                #if len(file)> 1:
                 #   label_filename = label_filename.split(" ")[0]
                    #print("update" , label_filename)
                    #sys.exit()
                if not os.path.exists(label_filename) or not os.path.exists(img_filename_r):
                    continue

                labels = open(label_filename, "r").readlines()
                for single_label in labels:
                    single_label_arr = single_label.split(" ")
                    f_label = [float(f) for f in single_label_arr]
                    f_label[0] = int(f_label[0])
                    types[f_label[0]] += 1
                    y.append(f_label[0])
                    X.append(f_label[1:])

        X = np.asarray(X)
        y = np.asarray(y)

        print ("CLASSES NUMBES {}".format(types))
        return X,y

def special_metric(y_valid, y_pred):
    assert len(y_valid) == len(y_pred)
    counter = 0
    for a,b in zip(y_valid, y_pred):
        if np.fabs(a-b)>1:
            counter +=1.0
    print ("special metric to be Updated ", counter/len(y_valid))

def special_metric_v2(y_valid, y_pred):
    assert len(y_valid) == len(y_pred)
    WEIGHTS=np.array(([0,4,4,4],[2,0,3,4],[1,2,0,3],[1,1,2,0]))
    CLASS_COUNTERS = np.zeros(5)
    counter = list()
    for a,b in zip(y_valid, y_pred):
        counter.append(WEIGHTS[a,b])
        CLASS_COUNTERS[WEIGHTS[a,b]]+=1
    print ("average speacial metric ", np.mean(counter))
    print ("Max special metric ", np.max(counter))
    print ("Min special metric ", np.min(counter))
    print ("CLASS COUNTERS ", CLASS_COUNTERS)

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
    print ("TRAIN FILEPATH ", train_filepath)
    valid_filepath = os.path.join(rootpath, ddict["val"])
    test_filepaht = os.path.join(rootpath, ddict["test"])

    X_train, y_train = read_docs(train_filepath,20)
    X_valid, y_valid = read_docs(valid_filepath,20)
    print ("Train size {} ".format(X_train.shape))
    savename = sys.argv[2]
    my_logger = get_logger()


    for mode in ["logistics", "bayes", "svc"]:
        my_logger.info("MODE " + mode)
        if mode == "logistics":
            model = LogisticRegression(solver='liblinear', random_state=0)
        if mode == "bayes":
            model = GaussianNB()
        if mode == "svc":
            model = SVC(class_weight="balanced")
        model.fit(X_train,y_train)
        print ("train score ", model.score(X_train, y_train))
        print ("valid score ", model.score(X_valid, y_valid))
        y_pred = model.predict(X_valid)
        print ("mean absolute_error: ", mean_absolute_error(y_valid, y_pred))
        special_metric(y_valid, y_pred)
        special_metric_v2(y_valid, y_pred)

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
        plt.savefig("cm_{}_{}.jpg".format(mode,savename))
        plt.show()

        print(classification_report(y_valid, model.predict(X_valid)))
