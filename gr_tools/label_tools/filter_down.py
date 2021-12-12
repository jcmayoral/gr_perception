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
            labels[key1][key2] = os.path.join(root, f)
    return labels

labels_folder='/home/jose/datasets/KITTI/training/image_02/'
labels_dict=load_labels(labels_folder)

#print(labels_dict, labels_folder)
with open (sys.argv[1], 'r') as imgs:
    for i in imgs:
        index = i.rstrip().split("/")
        key1 = index[-2]
        key2 = index[-1].split(".")[0]
        print(key1, key2)
        if key1 in labels_dict.keys():
            print (key1, "found")
            if key2 in labels_dict[key1].keys():
                print(key2, "found")
                print(labels_dict[key1][key2])
        continue
        if index in labels_dict.keys():
           # print('index', index)
            lfile = labels_dict[index]
            with open(lfile, 'r') as file:
                for label_raw in file:
                    cl, xc,yc,xw,yh = label_raw.rstrip().split(" ")
                    #print(cl, i, "AAA")
        continue
        img = cvw.imread(i)
