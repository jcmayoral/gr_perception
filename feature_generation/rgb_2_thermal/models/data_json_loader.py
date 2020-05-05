import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import cv2
import random
from sklearn import preprocessing
import json

class DataJSONLoader():
    def __init__(self, dataset_name, img_res=(128, 128),
                 rgb_dataset_folder="/floyd/input/flir_adas/train/RGB",
                 json_file="/floyd/input/flir_adas/train/thermal_8_bit", data_percentage=100):
        self.dataset_name = dataset_name
        self.data_percentage = data_percentage/100.
        print "data percentage ", self.data_percentage
        self.img_res = img_res
        self.thermal_res = (img_res[0], img_res[1],1)
        self.rgb_dataset_folder=rgb_dataset_folder

        with open(json_file) as f:
            self.json_file=json.load(f)

        #che check
        self.match_thermalfunction = self.match_by_name

        self.dyn_threshold = 0

        #TODO match images
        self.rgb_images_list=[image_name for image_name in os.listdir(self.rgb_dataset_folder) if image_name.endswith("jpg")]


    def load_samples(self,num_imgs=32,thermal_ext=".tiff"):
        rgb_imgs,thermal_imgs=[],[]
        random_rgb_image_name_list=np.random.choice(self.rgb_images_list,size=num_imgs).tolist()
        #random_thermal_image_name_list=np.random.choice(self.thermal_images_list,size=num_imgs).tolist()
        random_thermal_image_name_list = self.match_thermalfunction(random_rgb_image_name_list)

        for rgb_img_name, thermal_img_name in zip(random_rgb_image_name_list, random_thermal_image_name_list):
            rgb_img_path= os.path.join(self.rgb_dataset_folder,rgb_img_name)
            rgb_img=self.imread(rgb_img_path)
            thermal_img= self.thermal_imread(thermal_img_name.split(".")[0], rgb_img.shape)

            #rgb_img = cv2.resize(rgb_img, (self.img_res[0], self.img_res[1]))
            #rgb_img = rgb_img[:,:,np.newaxis]
            #thermal_img = cv2.resize(thermal_img, (self.img_res[0], self.img_res[1]))
            #r,thermal_img = cv2.threshold(thermal_img,127,1,cv2.THRESH_BINARY)

            rgb_img = rgb_img[:,:,np.newaxis]
            thermal_img = thermal_img[:,:,np.newaxis]
            rgb_imgs.append(rgb_img)
            thermal_imgs.append(thermal_img)
        rgb_imgs=np.array(rgb_imgs)#/127.5-1
        thermal_imgs=np.array(thermal_imgs)
        return rgb_imgs, thermal_imgs

    def match_by_name(self,rgb_files):
        return np.asarray(rgb_files)

    #practicaly the same of original_load_batch only difference on return
    def generator(self, batch_size=2, thermal_ext=".tiff"):
        self.n_batches = int(self.data_percentage*len(self.rgb_images_list) / batch_size)
        #self.n_batches = int(len(self.rgb_images_list) / batch_size)
        i = 0
        while True:
            batch = self.rgb_images_list[i*batch_size:(i+1)*batch_size]
            thermal_batch = self.match_thermalfunction(batch)

            i+=1
            if i == self.n_batches:
                print "RESTARTING GENERATOR"
                i = 0
            rgb_imgs, thermal_imgs = [], []
            for img_name, thermal_name in zip(batch, thermal_batch):
                rgb_img = self.imread(os.path.join(self.rgb_dataset_folder,img_name))
                thermal_img= self.thermal_imread(thermal_name.split(".")[0], rgb_img.shape)

                #rgb_img = cv2.resize(rgb_img, (self.img_res[0],self.img_res[1]))
                #thermal_img = cv2.resize(thermal_img, (self.img_res[0],self.img_res[1]))

                #r,thermal_img = cv2.threshold(thermal_img,127,1,cv2.THRESH_BINARY)

                if np.random.random() > 0.5:
                        rgb_img = np.fliplr(rgb_img)
                        thermal_img = np.fliplr(thermal_img)
                rgb_img = rgb_img[:,:,np.newaxis]
                thermal_img = thermal_img[:,:,np.newaxis]
                rgb_imgs.append(rgb_img)
                thermal_imgs.append(thermal_img)

            rgb_imgs = np.array(rgb_imgs)#[:,:,:]#/127.5 - 1.
            thermal_imgs = np.array(thermal_imgs)#[:,:,:]#/127.5 - 1
            yield rgb_imgs, thermal_imgs

    def rotate_image(self, image):
        (rows,cols) = image.shape
        return skimage.transform.rotate(image,180)#cv2.warpAffine(image, M, (rows, cols))

    def imread(self, path):
        img= cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def thermal_imread(self,img_id, rgb_shape):
        thermal_img = np.zeros(rgb_shape)
        needle = int(img_id.split("_")[1])
        for i in self.json_file['annotations']:
            if int(i["image_id"]) == needle:
                for x in range(i["bbox"][0], i["bbox"][0] + i["bbox"][2]):
                    for y in range(i["bbox"][1], i["bbox"][1]+ i["bbox"][3]):
                        thermal_img[x,y] = 255
        return thermal_img
