#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

def plot_bbs(image, bbs, visualize=False):
    height, width, channels = image.shape
    cll, cx1, cy1, cwidth, cheight  = bbs
    #x->width y->height
    x1 = int(np.rint((cx1 - cwidth/2)*width))
    y1 = int(np.rint((cy1 - cheight/2)*height))
    x2 = int(np.rint((cx1 + cwidth/2)*width))
    y2 = int(np.rint((cy1 + cheight/2)*height))
    #print (x1,y1)
    #print (x2,y2)
    #print (cll, type(cll))
    # Create a Rectangle patch
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image, "RING_" +str(int(cll)), (int(cx1*width),int(cy1*height)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)

    if visualize:
        cv2.imshow("TEST",image)
        cv2.waitKey()

def create_labels(store_path, labels_filepath):
    os.chdir(store_path)

    for root,dirs,files in os.walk(labels_filepath):
        print "LABEL ", root
        print "DIRS", dirs

        for file in files:
            label_filepath = os.path.join(root,file)
            print "LABELFILE ", label_filepath
            newfolder = file.split(".")[0]
            print newfolder
            try:
                os.mkdir(newfolder)
            except:
                print "Some abels exists delete or check them"
                sys.exit()

            with open(label_filepath, "r") as f:
                for line in f:
                    print line
                    img_id = line.rstrip().split()[0]
                    print img_id
                    #print data
                    #data = [float(d) for d in data
                    lfile = os.path.join(newfolder, img_id+".txt")
                    with open(lfile, "a") as lf:
                        lf.write("".join(line+"\n"))

if __name__ == "__main__":
    img_filepath =  "/home/jose/media/datasets/KITTI/data_tracking_image_2/training/image_02"
    labels_filepath = "/home/jose/media/datasets/KITTI/data_tracking_label_2/training/label_02"
    store_path = "/home/jose/media/datasets/new_kitti_labels"

    if len(sys.argv) == 1:
        print "missing instruction"
        sys.exit(0)

    if sys.argv[1] == "labels":
        create_labels(store_path, labels_filepath)
    counter = [0,0,0,0]
    aaa
    for img_folders in tqdm(os.walk(img_filepath)):
        print "A" , img_folders[1]
        for img_folder in img_folders[1]:
            print "B ", img_folder
            for img_file in img_folder:
                print "C", img_file


            print "AAA", os.path.file(img_folder[0])
            mages = open(filepath,'r')
            aaa

            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            labels = []
            if not os.path.exists(label_filename):
                print "label file {} not exists".format(label_filename)
                continue

            with open(label_filename, "r") as fl:
                labels = [data.strip().split(" ") for data in fl]#)
            fl.close()
            #print (label)
            img = cv2.imread(img_filename.rstrip().replace("txt", "jpg"))#, cv2.IMREAD_GRAYSCALE)

            for l in labels:
                detections = [float(d) for d in l]
                plot_bbs(img, detections, visualize = False, out=out)
                cl_ = int(detections[0])
                if cl_ < 0 or  cl_ > 3:
                    print "ERRROR.....",cl_, img_filename
                    continue
                counter[cl_] +=1
            #print("NEXT")
        #f.close()
        print "FINAL COUNTER", counter
