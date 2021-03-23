#!/usr/bin/python
from custom.evaluator import ImageEvaluator
import numpy as np
import rospy
import sys
import os
from tqdm import tqdm
import cv2

def match_bounding_boxes(gts, darknet_bbs,img_shape):
    #normalize measured
    height, width, channels = img_shape
    map_classes = {'lethal':0, 'danger': 1, 'warning':2, 'safe': 3}
    normalize_darknet = []
    for dbb in darknet_bbs:
        template_bb = [0,0,0,0,0]
        #Class goes init
        template_bb[0] = map_classes[dbb.Class]
        #xrange
        rx = dbb.xmax - dbb.xmin
        #xcenter
        cx = float(rx/2+ dbb.xmin)/width
        #yrange
        ry = dbb.ymax - dbb.ymin
        #ycenter
        cy = float(ry/2 + dbb.ymin)/height
        #update
        template_bb[1] = float(cx)
        template_bb[2] = float(cy)
        template_bb[3] = float(rx)/width
        template_bb[4] = float(ry)/height
        normalize_darknet.append(template_bb)

    #match best normalize_darknet to gt ..
    #assuming erros in classification
    pair_matches =[]
    indexes_not_available =[]

    for i,bb in enumerate(normalize_darknet):
        features = [0,0,0]
        features[0] = float(bb[3])*float(bb[4])#area
        features[1] = float(bb[1]) #cx
        features[2] = float(bb[2]) #cy

        matching_scores = []
        original_indexes =  []
        gfeatures=[0,0,0]
        #iterate in all gts
        for j,gbb in enumerate(gts):
            gfeatures[0] = float(gbb[3])*float(gbb[4])#area
            gfeatures[1] = float(gbb[1]) #cx
            gfeatures[2] = float(gbb[2]) #cy
            #calculate minimum score
            original_indexes.append(j)
            matching_scores.append(sum([abs(ca-cb) for ca,cb in zip(features,gfeatures)]))

        match_index = None
        #while check if indexes_not_available is less that the gts size
        flag = True
        while len(indexes_not_available) < len(gts) and len(matching_scores)>0 and flag:
            #get best score indez
            match_index = np.argmin(matching_scores)
            #if index has not been assigned
            if original_indexes[match_index] not in indexes_not_available:
                pair_matches.append([i, original_indexes[match_index]])
                indexes_not_available.append(original_indexes[match_index])
                flag = False
            else:
                matching_scores.remove(np.min(matching_scores))
                del original_indexes[match_index]
                match_index = None

    return pair_matches

if __name__ == "__main__":
    rospy.init_node('evaluator')
    rootpath = sys.argv[1]
    train_filepath = os.path.join(rootpath, "files_train.txt")
    valid_filepath = os.path.join(rootpath, "files_valid.txt")

    proc = ImageEvaluator()
    mean_absolute_error = 0
    fatal_misclassifications = 0
    total_images = 0

    if os.path.exists(valid_filepath):
        images = open(valid_filepath,'r').readlines()
        total_images = len(images)
        print "number of files {}".format(len(images))
        with tqdm(total=len(images)) as pbar:
            for img_filename in images:
                label_filename = img_filename.replace(".jpg", ".txt").rstrip()
                if not os.path.exists(label_filename):
                    print "label file {} not exists".format(label_filename)
                    continue

                gt_labels = []
                with open(label_filename, "r") as fl:
                    labels = [data.strip().split(" ") for data in fl]#)
                    gt_labels.append(labels[0])
                fl.close()
                #print (label)
                img = cv2.imread(img_filename.rstrip())#, cv2.IMREAD_GRAYSCALE)
                print type(img)
                #call darknet
                darknet_results = proc.darket_call(img)
                measured_labels = darknet_results.bounding_boxes.bounding_boxes
                sorted(measured_labels)
                #tuple of indexes [darknet, gt]
                matches = match_bounding_boxes(gt_labels, measured_labels, img.shape)

                for match in matches:
                    a = measured_labels[match[0]]
                    b = gt_labels[match[1]]
                    print "detected {} groud truth{}".format(a,b)
                    mean_absolute_error += abs(a-b)
                    if (abs(a-b)>1):
                        fatal_misclassifications+= 1
                pbar.update(1)

    print "MAE ", mean_absolute_error/total_images
    print "Percentual Fatal Misclassifications ", fatal_misclassifications/total_images
