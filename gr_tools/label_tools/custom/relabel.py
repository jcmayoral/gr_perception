#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm

def plot_bbs(image, bbs, require_new):
    height, width, channels = image.shape
    cll, cx1, cy1, cwidth, cheight  = bbs
    #x->width y->height
    x1 = int(np.rint((cx1 - cwidth/2)*width))
    y1 = int(np.rint((cy1 - cheight/2)*height))
    x2 = int(np.rint((cx1 + cwidth/2)*width))
    y2 = int(np.rint((cy1 + cheight/2)*height))
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image, str(int(cll)), (int(cx1*width),int(cy1*height)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)

    cv2.imshow("TEST",image)

    if require_new:
        cv2.putText(image, "Replace ", (int(cx1*width),20+int(cy1*height)), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255),2)
        cv2.imshow("TEST",image)
        key = cv2.waitKey(0)
        cv2.waitKey(1000)
        return key
    cv2.waitKey(50)

    return None


import fileinput
import time

def replace_line(file_name, line_nums, texts):
    lines = open(file_name, 'r').readlines()
    out = open(file_name, 'w')
    for line_nums, texts in zip(line_nums, texts):
        lines[line_nums] = texts
    out.writelines(lines)
    out.close()

def get_closest_v2(detections, ccs,fl):
    minval = 1000
    if len(fl) <4:
        return len(fl)
    minindex = 0
    for index,cc in enumerate(ccs):
        print cc
        score = np.sqrt(np.power(cc[0]-detections[3],2)+np.power(cc[1]-detections[4],2))
        #score = np.fabs(cc-detections[0])
        print score , "    SCORE "
        if index not in fl:
            if score < minval:
                minval = score
                minindex = index
    return minindex


def get_closest(detections,ccs):
    scores = []
    for cc in ccs:
        scores.append(np.sqrt(np.power(cc[0]-detections[1],2)+np.power(cc[1]-detections[2],2)))
    print "SCORES::::", scores
    if np.min(scores)> 0.1:
        return -1
    return np.argmin(scores)

def get_scores(detections,ccs):
    scores = []
    for cc in ccs:
        scores.append(np.sqrt(np.power(cc[0]-detections[1],2)+np.power(cc[1]-detections[2],2)))
    return scores

def get_score(detections,cc):
    return np.sqrt(np.power(cc[0]-detections[1],2)+np.power(cc[1]-detections[2],2))

if __name__ == "__main__":
    filepath =  "/home/jose/datasets/dummy/dummy"

    print "WAIT 2 seconds before start"
    time.sleep(2)
    print "OK"

    order_index = [-1,-1,-1,-1]
    cc = [[1000,1000],[1000,1000],[1000,1000],[1000,1000]]

    if os.path.exists(filepath):
        images = open(filepath,'r')

        start_index = 0

        if os.path.exists("lastindex.txt"):
            read_mode = "r"
            start_index = int(open("lastindex.txt",read_mode).readline().rstrip())
            print "start index"


        lastindex_file = open("lastindex.txt","w")
        skipped = open("skipped.txt","a")

        nflags = []

        for img_index, img_filename in tqdm(enumerate(images)):
            if img_index <= start_index:
                continue

            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            labels = []
            if not os.path.exists(label_filename):
                print "label file {} not exists".format(label_filename)
                continue

            lindx = 0

            #with  as fl:
            #for line in fileinput.input(label_filename, inplace=True):
            #print line
            replace_ind = []
            replace_data = []
            labels = open(label_filename, "r").readlines()

            for index, line in enumerate(labels):
                img = cv2.imread(img_filename.rstrip())
                label = [data for data in line.strip().split(" ")]#)
                detections = [float(d) for d in label]

                flag = False
                lindx = index

                #if len(labels)>1:
                #    lindx = get_closest_v2(detections, cc, nflags)
                if lindx not in nflags:
                    nflags.append(lindx)
                #print "\n" + "NFLAGS", nflags
                #print "\n" + "CCS ", cc
                #print "\n" + "order ", order_index
                #print "INDEX ", str(lindx)+"\n"
                #print "index  ", lindx, " Expected ", order_index[lindx], " GOTTEN ", detections[0]

                if np.fabs(int(detections[0]) - order_index[lindx])>0 and get_score(detections, cc[lindx])>0.1:
                    flag = True

                #print detections[0]
                #print "INDEX ", order_index


                key = plot_bbs(img, detections, flag)

                if key == None:
                    order_index[lindx] = int(label[0])
                    cc[lindx] = [float(detections[3]), float(detections[4])]
                    #lindx = lindx + 1
                    continue
                if key == 27:
                    lastindex_file.close()
                    sys.exit()

                label[0] = key - 176

                order_index[lindx] = int(label[0])
                cc[lindx] = [float(detections[3]), float(detections[4])]

                newlabel = ""
                newlabel = newlabel.join([str(c)+ " " for c in label])[:-1]+"\n"

                replace_ind.append(index)
                replace_data.append(newlabel)

                lastindex_file.seek(0)
                lastindex_file.write(str(img_index))

                #replace_line(label_filename,replace_ind,replace_data)

        lastindex_file.close()
