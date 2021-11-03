#!/usr/bin/python3
import cv2
import os
import sys
import numpy as np
import tqdm

def plot_bbs(image, bbs, visualize=False, out=None):
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
        cv2.waitKey(25)

    if out is not None:
        out.write(image)


if __name__ == "__main__":
    filepath =  sys.argv[1] #"/media/datasets/simanimation/depth_testdataset_v3/files.txt"
    print(filepath)
    counter = [0,0,0,0]
    opencounter = 0
    plot_fig = bool(int(sys.argv[2]))
    if plot_fig:
        #cap = cv2.VideoCapture('chaplin.mp4')
        out = cv2.VideoWriter('dataset.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

    if os.path.exists(filepath):
        print ("FILES_>",  sum(1 for line in open(filepath)))
        images = open(filepath,'r')
        for img_filename in tqdm.tqdm(images, ascii=True, desc="plot images"):
            #if opencounter%1000== 0:
            #    print (opencounter, counter)
            #    print (img_filename)
            opencounter = opencounter+1
            label_filename = img_filename.replace(".jpg", ".txt").rstrip()
            if not os.path.exists(label_filename):
                continue
            fl = open(label_filename, "r")
            label = fl.readline().split(" ")
            fl.close()
            detections = [float(d) for d in label]

            if plot_fig:
                img = cv2.imread(img_filename.rstrip())#, cv2.IMREAD_GRAYSCALE)
                # out.write(img)
                plot_bbs(img, detections, visualize = True, out=out)
            cl_ = int(detections[0])
            if cl_ < 0 or  cl_ > 3:
                print (cl_, img_filename)
                continue
            counter[cl_] = counter[cl_] + 1
            #print ("counters ", counter) 
            #print("NEXT")
        print ("closing main file")
        images.close()

    if plot_fig:
        out.release()

    print ("FINAL COUNT", counter)
