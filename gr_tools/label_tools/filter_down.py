import cv2
import sys
import os

try:
    os.mkdir("output")
except:
    print("folder output exists")

def load_labels(labels_folder='labels'):
    labels = dict()
    for root, folder, file in os.walk(labels_folder):
        for f in file:
            if "raw" not in f:
                continue
            key1 = root.split("/")[-1]
            key2 = f.split(".")[0]
            if key1 not in labels:
                labels[key1] = dict()
            labels[key1][key2] = os.path.join(root, f).rstrip()
    return labels

labels_folder='/home/jose/datasets/KITTI/training/image_02/'
#Load labels on validation data
labels_dict=load_labels(labels_folder)

filter_threshold = 15.0

outputfilename = os.path.join("output", "filtered_valid.txt")
outputfile = open(outputfilename, "a")
#print(labels_dict, labels_folder)
with open (sys.argv[1], 'r') as imgs:
    for i in imgs:
        index = i.rstrip().split("/")
        key1 = index[-2]
        key2 = index[-1].split(".")[0]
        if key1 in labels_dict.keys():
            if key2 in labels_dict[key1].keys():
                lfile = labels_dict[key1][key2]

                #If true store
                flag = True
                with open(lfile, 'r') as file:
                    for label_raw in file:
                        cl, xc,yc,xw,yh = label_raw.rstrip().split(" ")
                        print(cl, i, "AAA")
                        if float(cl)> filter_threshold:
                            flag=False
                if (flag):
                    print("store", lfile, i)
                    outputfile.write(i)
        continue
        img = cvw.imread(i)
outputfile.close()
