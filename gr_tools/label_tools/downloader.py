import cv2
import sys
import os
from tqdm import tqdm

foldername="validation_images"

try:
    os.mkdir(foldername)
except:
    print("folder output exists")

try:
    os.chdir(foldername)
except:
    print("change to folder "+foldername)

with open (sys.argv[1], 'r') as imgs:
    for i in tqdm(imgs):
        filename = i.split("/")[-1].rstrip()
        cv2.imwrite(filename,cv2.imread(i.rstrip()))