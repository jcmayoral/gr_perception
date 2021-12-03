#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#plt.ion()

RANGES = dict()

def plot_bbs(image,cll,dclass,x1,y1,x2,y2):
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image,"{}_{}".format(cll, dclass), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),1)
    #if visualize:
    #    cv2.imshow("TEST",image)
    #cv2.waitKey(10)

def dynamic_class(distance):
    ranges = np.array([[0,8],[8,12], [12,19], [19,120]])
    for cl, limits in enumerate(ranges):
        if distance >= limits[0] and distance < limits[1]:
            print(cl)
            return cl


def visualize(img_filepath, labels_filepath, classification_distance = 10.0):
    os.chdir(img_filepath)
    classes = dict()
    x = list()
    print("classification distance ", classification_distance)

    for root,dirs,files in os.walk("."):
        #print ("LABEL ", root)
        #print ("DIRS", dirs)
        #print "FILES", files
        for file in files:
            if "png" not in file:
                continue
            labelfile = os.path.join(labels_filepath, root.split("/")[1],str(int(file.split(".png")[0]))+".txt")
            if not os.path.exists(labelfile):
                continue

            img_file = os.path.join(root,file)
            cv_img = cv2.imread(img_file)

            with open(labelfile, "r") as fl:
                for line in fl:
                    #print line
                    data = line.rstrip().split(" ")
                    #print data, data[0]
                    dclass = dynamic_class(float(data[15]))#int(float(data[15])/ classification_distance)
                    #if dclass > 3:
                    #    continue
                    #dclass = data[15]
                    if float(data[15]) < 0:
                            continue

                    if data[2] in classes.keys():
                        classes[data[2]][dclass] = classes[data[2]][dclass]+ 1
                    else:
                        classes[data[2]] = np.zeros(50)
                        classes[data[2]][dclass] = classes[data[2]][dclass] + 1

                    x.append(float(data[15]))
                    bbs = [int(float(bb)) for bb in data[6:10]]
                    plot_bbs(cv_img, data[2],dclass, *bbs)

                    #if dclass == 0:
                    #    print data[2], dclass, float(data[15])
                    #    cv2.waitKey(0)

            cv2.imshow("visualize", cv_img)
            cv2.waitKey(25)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bins = [0, classification_distance, classification_distance*2, classification_distance*3, classification_distance*4]
    bins = 4
    grid_x_ticks = np.arange(0, 75.0, classification_distance)
    grid_y_ticks = np.arange(0, 1.0, 0.05)

    ax.set_xticks(grid_x_ticks, minor=True)
    ax.set_yticks(grid_y_ticks, minor=True)
    #plt.grid(True, color="grey", linewidth="1.4", linestyle="-.")
    ax.grid(which='both')
    ax.grid(which='minor', alpha=1.0, linestyle='-.')
    (n, bins, patches) = plt.hist(x, bins=100, cumulative=True, weights=np.ones(len(x)) / len(x))
    plt.savefig("/home/jose/fig1.png")
    print("total", sum(n))
    print(n,bins,patches)
    """
    thres = int(sum(n/4))
    foundt = list()

    cum = 0
    for i in n:
        cum +=i
        if (cum > thres):
            print ("threshold ", n)
            foundt.append(n)
            thres = thres + int(sum(n)/4)

    #bins = [0, classification_distance, classification_distance*2, classification_distance*3, classification_distance*4]
    bins = 100
    (n, bins, patches) = plt.hist(x, bins=bins, cumulative=False)
    """

    #plt.plot(np.arange(10), np.arange(10))
    fig2 = plt.figure()
    (n2, bins2, patches2) = plt.hist(x, bins=100, cumulative=False)
    plt.savefig("/home/jose/fig2.png")
    plt.show()
    print ("classes summary")
    for i, j in classes.items():
        print ("class {}  count {}".format(i,j))


def create_labels(img_filepath, labels_filepath, classification_distance = 10.0):
    masterfile_name = "images_collection.txt"

    main_file = os.path.join(img_filepath,masterfile_name)
    os.chdir(img_filepath)
    classes = dict()
    x = list()

    for root,dirs,files in tqdm(os.walk(".")):
        #print "LABEL ", root
        #print "DIRS", dirs
        #print ("FILES", files)
        for file in tqdm(files):
            if not "png" in file:
                print ("ignore ", file)
                continue
            labelfile = os.path.join(labels_filepath, root.split("/")[1],str(int(file.split(".png")[0]))+".txt")
            #print labelfile
            if not os.path.exists(labelfile):
                continue

            img_file = os.path.join(root,file)
            newlabel_file = img_file.replace("png", "txt")

            cv_img = cv2.imread(img_file)
            flag = False

            with open(labelfile, "r") as fl:
                for line in fl:
                    #print line
                    data = line.rstrip().split(" ")
                    #print data, data[0]
                    dclass = dynamic_class(float(data[15]))#min(3, int(float(data[15])/ classification_distance))
                    #dclass = data[15]
                    if float(data[15]) < 0:
                        continue

                    flag = True

                    if data[2] in classes.keys():
                        classes[data[2]][dclass] = classes[data[2]][dclass]+ 1
                    else:
                        classes[data[2]] = [0,0,0,0,0,0,0,0,0,0]
                        classes[data[2]][dclass] = classes[data[2]][dclass] + 1
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
    print ("classes summary")
    for i, j in classes.items():
        print ("class {}  count {}".format(i,j))

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
        print ("LABEL ", root)
        print ("DIRS", dirs)

        for file in files:
            label_filepath = os.path.join(root,file)
            print ("LABELFILE ", label_filepath)
            newfolder = file.split(".")[0]
            print ("folder", newfolder)
            try:
                os.mkdir(newfolder)
            except:
                print ("Some labels exists delete or check them")
                sys.exit()

            with open(label_filepath, "r") as f:
                for line in f:
                    idclass = line.rstrip().split()[2]
                    if idclass != "Pedestrian" and idclass != "Cyclist":
                        print("skip ", idclass)
                        continue
                    img_id = line.rstrip().split()[0]
                    #print data
                    #data = [float(d) for d in data
                    lfile = os.path.join(newfolder, img_id+".txt")
                    with open(lfile, "a") as lf:
                        lf.write("".join(line))

if __name__ == "__main__":
    img_filepath =  "/home/jose/datasets/KITTI/training/image_02"
    labels_filepath = "/home/jose/datasets/KITTI/training/label_02"
    store_path = "/home/jose/datasets/new_kitti_labels"

    if len(sys.argv) == 1:
        print ("missing instruction")
        sys.exit(0)

    if sys.argv[1] == "split":
        split_labels(store_path, labels_filepath)

    if sys.argv[1] == "visualize":
        visualize(img_filepath, store_path, classification_distance=float(sys.argv[2]))

    if sys.argv[1] == "create":
        create_labels(img_filepath, store_path, classification_distance=float(sys.argv[2]))

    if sys.argv[1] == "raw_labels":
        create_raw_labels(img_filepath, store_path)
