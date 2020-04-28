import cv2
from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from data_loader import DataLoader
from unet import  sample_test_images
import pickle
import os

def filter_thermal(original_thermal):
    proccessed_images = original_thermal
    for i in original_thermal:
        print "image proccessed"
    return proccessed_images


if __name__ == '__main__':
    batch_size = 5
    im_size = (128,128)
    dataset_name = "flir_8"
    data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1]),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False)


    rgb_images, thermal_images = data_loader.load_samples(num_imgs=batch_size, thermal_ext=".jpeg")
    print rgb_images[0].shape
    processed_imgs_thermal = filter_thermal(thermal_images)
    sample_test_images(rgb_images, processed_imgs_thermal, batch_size)
