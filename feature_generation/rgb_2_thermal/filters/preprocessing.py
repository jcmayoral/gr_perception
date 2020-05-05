import sys
sys.path.append("..")

from tools.tools import visualize
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from skimage import img_as_float

#https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()
    ###
    #a = (target_type_max - target_type_min) / (imax - imin)
    #b = target_type_max - a * ima
    #new_img = (a * img + b).astype(target_type)
    #new_img = img/imax
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    #print np.unique(new_img , return_counts=True), " AKA "
    #new_img = (new_img * 255).astype(target_type)
    return new_img

def filter_thermal(original_thermal,threshold = 180):
    proccessed_images = list()
    for i,im in enumerate(original_thermal):
        original_thermal[i] = convert(im, 0, 255, np.uint8)
    #print np.unique(original_thermal[0])
    return original_thermal, threshold

    for i in original_thermal:
        j = convert(i, 0, 255, np.uint8)
        #j =  img_as_float(i)
        #print mean
        fthreshold = threshold
        #print np.unique(j)
        #print original_thermal.shape
        #print fthreshold

        if threshold < 10:
            #print "hack" , threshold
            fthreshold = np.mean(j)+np.sqrt(np.var(j))#threshold


        h = j.shape[0]
        w = j.shape[1]

        if 0 > threshold > 255:
            print "ERROR ", threshold

        for y in range(0, h):
            for x in range(0, w):
                j[y, x] = 0 if j[y, x] > fthreshold else 1
        #j = convert(i, 0, 255, np.uint8)
        #print np.unique(j, return_counts = True), threccshold
        #print threshold
        #if i.dtype!=np.uint8:
        #    print "converting", j.dtype
        #    j = convert(j, 0, 255, np.uint8)

        #print np.unique(j, return_counts=True)
        #
        #j = cv2.dilate(j, (3,3), iterations=1)
        #i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        #j = cv2.Laplacian(j, cv2.CV_8UC1, ksize=3)
        #r,j = cv2.threshold(j,threshold,1,cv2.THRESH_BINARY)
        #print np.unique(j)
        #j = cv2.erode(j, (3,3), iterations=1)
        #print "image proccessed"
        proccessed_images.append(j)

    thermal = np.asarray(proccessed_images)[:,:,:]#,np.newaxis]
    return thermal, fthreshold

if __name__ == '__main__':
    batch_size = 5
    im_size = (128,128)
    dataset_name = "flir_8"
    from data_loader import DataLoader

    data_loader = DataLoader(dataset_name=dataset_name,
                        img_res=(im_size[0], im_size[1],1),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False)
    rgb_images, thermal_images = data_loader.load_samples(num_imgs=batch_size, thermal_ext=".jpeg")
    processed_imgs_thermal, threshold = filter_thermal(thermal_images)
