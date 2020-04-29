import cv2
import matplotlib.pyplot as plt
from tools import visualize
from unet import  sample_test_images
import pickle
import os
import numpy as np
#https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
def convert(img, target_type_min, target_type_max, target_type):
    imin = 0#img.min()
    imax = img.max()
    ###
    #a = (target_type_max - target_type_min) / (imax - imin)
    #b = target_type_max - a * ima
    #new_img = (a * img + b).astype(target_type)
    new_img = img.astype(np.float)/imax
    #print np.unique(new_img , return_counts=True), " AKA "
    new_img = (new_img * 255).astype(target_type)
    return new_img

def filter_thermal(original_thermal,threshold = 180):
    proccessed_images = list()
    for i in original_thermal:
        j = convert(i, 0, 255, np.uint8)
        #j = cv2.dilate(j, (3,3), iterations=1)
        #i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        #j = cv2.Laplacian(j, cv2.CV_8UC1, ksize=3)
        r,j = cv2.threshold(j,threshold,1,cv2.THRESH_BINARY)
        #print np.unique(j)
        #j = cv2.erode(j, (3,3), iterations=1)
        #print "image proccessed"
        proccessed_images.append(j)

    thermal = np.asarray(proccessed_images)[:,:,:,np.newaxis]
    return thermal

if __name__ == '__main__':
    batch_size = 5
    im_size = (128,128)
    dataset_name = "flir_8"
    from data_loader import DataLoader

    data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1]),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False)


    rgb_images, thermal_images = data_loader.load_samples(num_imgs=batch_size, thermal_ext=".jpeg")
    processed_imgs_thermal = filter_thermal(thermal_images)
    print thermal_images.shape, processed_imgs_thermal.shape
    sample_test_images(thermal_images, processed_imgs_thermal, batch_size)
