import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import cv2
import random
from sklearn import preprocessing

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128),
                 rgb_dataset_folder="/floyd/input/flir_adas/train/RGB",
                 thermal_dataset_folder="/floyd/input/flir_adas/train/thermal_8_bit",
                 path_timestamp_matching="~/", match_by_timestamps=False):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.thermal_res = (img_res[0], img_res[1],1)
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        self.path_timestamp_matching = path_timestamp_matching

        if match_by_timestamps:
            self.match_thermalfunction= self.match_by_timestamp
        else:
            self.match_thermalfunction = self.match_by_name

        #TODO match images
        self.thermal_min_max_scaler = preprocessing.MinMaxScaler()
        self.rgb_images_list=[image_name for image_name in os.listdir(self.rgb_dataset_folder) if image_name.endswith("jpg")]
        self.thermal_images_list=[image_name for image_name in os.listdir(self.thermal_dataset_folder) if image_name.endswith("tiff")]

    def load_samples(self,num_imgs=32,thermal_ext=".tiff"):
        rgb_imgs,thermal_imgs=[],[]
        random_rgb_image_name_list=np.random.choice(self.rgb_images_list,size=num_imgs).tolist()
        #random_thermal_image_name_list=np.random.choice(self.thermal_images_list,size=num_imgs).tolist()
        random_thermal_image_name_list = self.match_thermalfunction(random_rgb_image_name_list)

        for rgb_img_name, thermal_img_name in zip(random_rgb_image_name_list, random_thermal_image_name_list):
            rgb_img_path= os.path.join(self.rgb_dataset_folder,rgb_img_name)
            thermal_img_path=os.path.join(self.thermal_dataset_folder,thermal_img_name.split(".")[0]+thermal_ext)
            rgb_img=self.imread(rgb_img_path)
            thermal_img=self.thermal_imread(thermal_img_path)
            rgb_img = cv2.resize(rgb_img, self.img_res)
            thermal_img = cv2.resize(thermal_img, self.img_res)
            rgb_imgs.append(rgb_img)
            thermal_imgs.append(thermal_img)
        rgb_imgs=np.array(rgb_imgs)/127.5-1
        thermal_imgs=np.array(thermal_imgs)[:,:,:,np.newaxis]/127.5-1
        return rgb_imgs, thermal_imgs

    def match_by_name(self,rgb_files):
        return np.asarray(rgb_files)

    #https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
    def convert(self,img, target_type_min, target_type_max, target_type):
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

    def match_by_timestamp(self, rgb_files):
        thermal_files = list()
        for r in rgb_files:
            r = r.replace("image_filter_bag","")
            r = r.replace(".jpg","")
            r = r[2:]
            filename = os.path.join(self.path_timestamp_matching, r +".txt")
            with open(filename) as f:
                 thermal_index = f.read()
            #TODO this is totally bruteforce
            #if not found set a default
            needle_file = self.thermal_images_list[-1]
            for image_file in self.thermal_images_list:
                if thermal_index in image_file:
                    #print "found ", image_file, thermal_index
                    needle_file = image_file
                    break;
            thermal_files.append(needle_file)
        return np.asarray(thermal_files)

    def load_batch(self, batch_size=1, is_testing=False,thermal_ext=".tiff"):
        self.n_batches = int(len(self.rgb_images_list) / batch_size)

        for i in range(self.n_batches-1):
            batch = self.rgb_images_list[i*batch_size:(i+1)*batch_size]
            thermal_batch = self.thermal_images_list[i*batch_size:(i+1)*batch_size]
            rgb_imgs, thermal_imgs = [], []
            for img_name, thermal_name in zip(batch, thermal_batch):
                rgb_img = self.imread(os.path.join(self.rgb_dataset_folder,img_name))
                rgb_img /= 255#self.rgb_min_max_scaler.fit_transform(rgb_img)
                #thermal_img= self.thermal_imread(os.path.join(self.thermal_dataset_folder,img_name.split(".")[0]+thermal_ext))
                thermal_img = self.thermal_imread(os.path.join(self.thermal_dataset_folder,thermal_name))
                #thermal_img = self.thermal_min_max_scaler.fit_transform(thermal_img)
                h, w, _ = rgb_img.shape
                rgb_img = cv2.resize(rgb_img, self.img_res)
                thermal_img = cv2.resize(thermal_img, self.img_res)
                thermal_img = thermal_img.reshape(self.thermal_res)

                if not is_testing and np.random.random() > 0.5:
                        rgb_img = np.fliplr(rgb_img)
                        thermal_img = np.fliplr(thermal_img)

                rgb_imgs.append(rgb_img)
                thermal_imgs.append(thermal_img)

            #standardize?
            rgb_imgs = np.array(rgb_imgs)
            #feature scaling...
            thermal_imgs = np.array(thermal_imgs)
            yield rgb_imgs, thermal_imgs

    #fieldsafe specific
    def rotate_image(self, image):
        (rows,cols) = image.shape
        #M = cv2.getRotationMatrix2D((rows/2, cols/2), 180, 1)
        #thermal_img = imutils.rotate(thermal_img, 180, dtype = np.float32)
        return skimage.transform.rotate(image,180)#cv2.warpAffine(image, M, (rows, cols))

    #fieldsafe specific
    def crop_image(self, image):
        return image[:,300:,:]

    def imread(self, path):
        try:
            img= cv2.imread(path)
            #For FIELDSAFE
            #img = self.crop_image(img)
            img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
        except:
            print(path)

    def thermal_imread(self,img_path):
        thermal_img_path= img_path
        thermal_img= skimage.io.imread(thermal_img_path)
        if thermal_img.dtype != np.uint8:
            thermal_img = self.convert(thermal_img, 0, 255, np.uint8)
            #print np.unique(thermal_img, return_counts=True)
            #ret,thermal_img = cv2.threshold(thermal_img,160,255,cv2.THRESH_BINARY)

        return thermal_img
