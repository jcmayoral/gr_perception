import os
import sys
from tqdm import tqdm

outputfile = open("simtotal.txt", "a")
trainfile = open("simtrain.txt", "a")
valfile = open("simval.txt", "a")
counter = 0

for root, folder, files in tqdm(os.walk(sys.argv[1])):
    for ffile in files:
        if "augment" in ffile and "jpg" in ffile:
            imagefile = os.path.join(root, ffile)
            labelfile = imagefile.replace("jpg", "txt")
            if os.path.exists(labelfile) and os.path.exists(imagefile):
                outputfile.write(imagefile+"\n")
                if counter%3 == 0:
                    valfile.write(imagefile+"\n")
                else:
                    trainfile.write(imagefile+"\n")
                counter+=1

outputfile.close()