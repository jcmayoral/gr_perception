import cv2
import sys
import os

try:
    os.mkdir("output")
except:
    print("folder output exists")

def load_labels(labels_folder='labels'):
    labels = list()
    for root, folder, file in os.walk(labels_folder):
        for f in file:
            print("f", f)
            labels.append(f.split(".")[0])
            labels.append(f.split(".")[0])
    return labels

labels_dict=load_labels()
print(labels_dict)

with open (sys.argv[1], 'r') as imgs:
    for i in imgs:
        index = i.split("/")[-1].rstrip().split(".")[0]
        if not index in labels_dict:
            print ("OK", i, index. labels_dict[index])
        labels_file = i.replace("png", "txt")
        continue
        with open(labels_file, 'r') as file:
            for f in file:
                print("f")
        continue
        img = cvw.imread(i)
