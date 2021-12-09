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
            labels.append(f)
    return labels

labels_dict=load_labels()
print (labels_dict)
sys.exit()

with open (sys.argv[1], 'r') as imgs:
    for i in imgs:
        img = cvw.imread(i)
