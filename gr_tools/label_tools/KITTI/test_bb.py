#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#plt.ion()

def plot_bbs(image,cll,dclass,x1,y1,x2,y2):
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image,"{}_{}".format(cll, dclass), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)

    #if visualize:
    #    cv2.imshow("TEST",image)
    #    cv2.waitKey()

def visualize(img_filepath, labels_filepath, classification_distance = 15.0):
    os.chdir(img_filepath)
    classes = dict()
    x = list()

    for root,dirs,files in os.walk("."):
        print "LABEL ", root
        print "DIRS", dirs
        #print "FILES", files
        for file in files:
            labelfile = os.path.join(labels_filepath, root.split("/")[1],str(int(file.split(".png")[0]))+".txt")
            print labelfile
            if not os.path.exists(labelfile):
                continue

            img_file = os.path.join(root,file)
            cv_img = cv2.imread(img_file)

            with open(labelfile, "r") as fl:
                for line in fl:
                    #print line
                    data = line.rstrip().split(" ")
                    #print data, data[0]
                    dclass = min(3, int(float(data[15])/ classification_distance))
                    #dclass = data[15]
                    if float(data[15]) < 0:
                        continue

                    if data[2] in classes.keys():
                        classes[data[2]][dclass] = classes[data[2]][dclass]+ 1
                    else:
                        classes[data[2]] = [0,0,0,0]
                        classes[data[2]][dclass] = classes[data[2]][dclass] + 1

                    x.append(float(data[15]))
                    bbs = [int(float(bb)) for bb in data[6:10]]
                    plot_bbs(cv_img, data[2],dclass, *bbs)

                    #if dclass == 0:
                    #    print data[2], dclass, float(data[15])
                    #    cv2.waitKey(0)

            cv2.imshow("visualize", cv_img)
            cv2.waitKey(25)
    plt.figure()
    plt.hist(x, bins=40, cumulative=False)
    #plt.plot(np.arange(10), np.arange(10))
    plt.show()
    print "classes summary"
    for i, j in classes.iteritems():
        print "class {}  count {}".format(i,j)


def create_labels(img_filepath, labels_filepath, classification_distance = 15.0, v2=False):
    masterfile_name = "v2_images_collection.txt"
    if not v2:
        masterfile_name = "images_collection.txt"

    main_file = os.path.join(img_filepath,masterfile_name)
    os.chdir(img_filepath)
    classes = dict()
    x = list()

    if v2:
        offsets_ = {"Person":0, "Pedestrian": 0, "Cyclist":0, "Van": 1, "Tram":1, "Car": 1, "Truck":1}

    for root,dirs,files in tqdm(os.walk(".")):
        #print "LABEL ", root
        #print "DIRS", dirs
        #print "FILES", files
        for file in tqdm(files):
            if not "png" in file:
                continue
            labelfile = os.path.join(labels_filepath, root.split("/")[1],str(int(file.split(".png")[0]))+".txt")
            #print labelfile
            if not os.path.exists(labelfile):
                continue

            img_file = os.path.join(root,file)
            newlabel_file = img_file.replace("png", "txt")
            #if v2:
            #    newlabel_file+="v2"
            cv_img = cv2.imread(img_file)
            flag = False

            with open(labelfile, "r") as fl:
                for line in fl:
                    #print line
                    data = line.rstrip().split(" ")
                    #print data, data[0]
                    dclass = min(3, int(float(data[15])/ classification_distance))
                    #dclass = data[15]
                    if float(data[15]) < 0:
                        continue

                    flag = True
                    if v2 and data[2] in offsets_.keys():
                        dclass = dclass + offsets_[data[2]]*4
                    if dclass>3:
                        print dclass, data[2]

                    if data[2] in classes.keys():
                        classes[data[2]][dclass] = classes[data[2]][dclass]+ 1
                    else:
                        classes[data[2]] = [0,0,0,0,0,0,0,0,0,0]
                        classes[data[2]][dclass] = classes[data[2]][dclass] + 1

                    x.append(float(data[15]))
                    bbs = [int(float(bb)) for bb in data[6:10]]
                    #plot_bbs(cv_img, data[2],dclass, *bbs)
                    create_label(newlabel_file, str(dclass), cv_img.shape, *bbs)

                    #if dclass == 0:
                    #    print data[2], dclass, float(data[15])
                    #    cv2.waitKey(0)
            if flag:
                with open(main_file, "a+") as mfile:
                    mfile.write( os.path.join(img_filepath,root[2:], file)+"\n")

            #cv2.imshow("visualize", cv_img)
            #cv2.waitKey(25)
    plt.figure()
    plt.hist(x, bins=40, cumulative=False)
    #plt.plot(np.arange(10), np.arange(10))
    plt.show()
    print "classes summary"
    for i, j in classes.iteritems():
        print "class {}  count {}".format(i,j)


def create_label(nlf, dclass,img_shape, x1,y1,x2,y2):
    height, width, channels = img_shape
    ring = dclass
    data = str(ring) + " "
    rx = int(x2-x1)
    cx = float(rx/2+ x1)/width
    ry = int(y2 -y1)
    cy = float(ry/2 + y1)/height
    data += str(cx) + " "
    data += str(cy) + " "
    data += str(float(rx)/width) + " "
    data += str(float(ry)/height) + "\n"
    bbsx =[1,cx,cy,float(rx)/width,float(ry)/height]

    with open(nlf, "a+") as text_file:
        text_file.write(data)

def split_labels(store_path, labels_filepath):
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
                        lf.write("".join(line))

if __name__ == "__main__":
    img_filepath =  "/home/jose/media/datasets/KITTI/data_tracking_image_2/training/image_02"
    labels_filepath = "/home/jose/media/datasets/KITTI/data_tracking_label_2/training/label_02"
    store_path = "/home/jose/media/datasets/new_kitti_labels"

    if len(sys.argv) == 1:
        print "missing instruction"
        sys.exit(0)

    if sys.argv[1] == "split":
        split_labels(store_path, labels_filepath)

    if sys.argv[1] == "visualize":
        visualize(img_filepath, store_path)

    if sys.argv[1] == "create":
        create_labels(img_filepath, store_path)

    if sys.argv[1] == "new_labels":
        create_labels(img_filepath, store_path, v2=True)
