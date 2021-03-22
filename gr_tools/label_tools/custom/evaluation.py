#!/usr/bin/python
from custom.image_processing import ImageEvaluator
import numpy as np

def match_bounding_boxes(gts, darknet_bbs,img_shape):
    #normalize measured
    height, width, channels = image_shape
    map_classes = {'Lethal':0, 'Danger': 1, 'Warning':2, 'Safe', 3}
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
        ry = obb.ymax - dbb.ymin
        #ycenter
        cy = float(ry/2 + dbb.ymin)/height
        #update
        template_bb[1] = float(cx)
        template_bb[2] = float(cy)
        template_bb[3[ = float(rx)/width
        template_bb[4] = float(ry)/height
        normalize_darknet.append(template_bb)

    #match best normalize_darknet to gt ..
    #assuming erros in classification
    pair_matches =[]
    indexes_not_available =[]

    for i,bb in enumerate(normalize_darknet):
        features = [0,0,0]
        features[0] = bb[3]*bb[4]#area
        features[1] = bb[1] #cx
        features[2] = bb[2] #cy

        matching_scores = []
        gfeatures=[0,0,0]
        #iterate in all gts
        for gbb in gts:
            gfeatures[0] = gbb[3]*gbb[4]#area
            gfeatures[1] = gbb[1] #cx
            gfeatures[2] = gbb[2] #cy
            #calculate minimum score
            matching_scores.append(sum([abs(ca-cb) for ca,cb in zip(features,gfeatures)]))

        match_index = None
        #while check if indexes_not_available is less that the gts size
        flag = True
        while len(indexes_not_available) < len(gts) and len(matching_scores)>0 and flag:
            #get best score indez
            match_index = np.argmin(matching_scores)
            #if index has not been assigned
            if match_index not in indexes_not_available:
                pair_matches.append([i, match_index])
                indexes_not_available.append(match_index)
                flag = False
            else:
                matching_scores.remove(np.min(matching_scores))
                match_index = None

    return pair_matches

if __name__ == "__main__":
    rootpath = sys.argv[1]
    train_filepath = os.path.join(rootpath, "files_train.txt")
    valid_filepath = os.path.join(rootpath, "files_valid.txt")

    proc = ImageEvaluator(matches, depth_camera_info)

    if os.path.exists(valid_filepath):
        images = open(valid_filepath,'r').readlines()
        print "number of files {}".format(len(images))
        with tqdm(total=len(images)) as pbar:
            for img_filename in images:
                label_filename = img_filename.replace(".jpg", ".txt").rstrip()
                if not os.path.exists(label_filename):
                    print "label file {} not exists".format(label_filename)
                    continue

                gr_labels = []
                with open(label_filename, "r") as fl:
                    labels = [data.strip().split(" ") for data in fl]#)
                    gr_labels.append(labels[0])
                fl.close()
                #print (label)
                img = cv2.imread(img_filename.rstrip())#, cv2.IMREAD_GRAYSCALE)
                #call darknet
                darknet_results = proc.darket_call(img)
                measured_labels = darknet_results.bounding_boxes.bounding_boxes
                sort(measured_labels)
                #tuple of indexes [darknet, gt]
                matches = match_bounding_boxes(gt_labels, measured_labels, img.shape)

                for match in matches:
                    a = measured_labels[match[0]][0]
                    b = gt_labels[match[1]][0]
                    print "detected {} groud truth{}".format(a,b)
                pbar.update(1)


    print "FINAL COUNTER", counter
