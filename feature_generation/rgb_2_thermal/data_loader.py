import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import cv2
import random

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128),
                 rgb_dataset_folder="/floyd/input/flir_adas/train/RGB",
                 thermal_dataset_folder="/floyd/input/flir_adas/train/thermal_8_bit"):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        self.rgb_images_list=[image_name for image_name in os.listdir(self.rgb_dataset_folder) if image_name.endswith("jpg")]

    def load_samples(self,num_imgs=32,thermal_ext=".tiff"):
        rgb_imgs,thermal_imgs=[],[]
        random_rgb_image_name_list=np.random.choice(self.rgb_images_list,size=num_imgs).tolist()
        for rgb_img_name in random_rgb_image_name_list:
            rgb_img_path= os.path.join(self.rgb_dataset_folder,rgb_img_name)
            thermal_img_path=os.path.join(self.thermal_dataset_folder,rgb_img_name.split(".")[0]+thermal_ext)
            rgb_img=self.imread(rgb_img_path)
            thermal_img=self.thermal_imread(thermal_img_path)
            rgb_img = cv2.resize(rgb_img, self.img_res)
            thermal_img = cv2.resize(thermal_img, self.img_res)
            rgb_imgs.append(rgb_img)
            thermal_imgs.append(thermal_img)
        rgb_imgs=np.array(rgb_imgs)/127.5-1
        thermal_imgs=np.array(thermal_imgs)[:,:,:,np.newaxis]/127.5-1
        return rgb_imgs, thermal_imgs

    def load_batch(self, batch_size=1, is_testing=False,thermal_ext=".tiff"):


        self.n_batches = int(len(self.rgb_images_list) / batch_size)

        for i in range(self.n_batches-1):
            batch = self.rgb_images_list[i*batch_size:(i+1)*batch_size]
            rgb_imgs, thermal_imgs = [], []
            for img_name in batch:
                rgb_img = self.imread(os.path.join(self.rgb_dataset_folder,img_name))
                thermal_img= self.thermal_imread(os.path.join(self.thermal_dataset_folder,img_name.split(".")[0]+thermal_ext))
                h, w, _ = rgb_img.shape

                rgb_img = cv2.resize(rgb_img, self.img_res)
                thermal_img = cv2.resize(thermal_img, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        rgb_img = np.fliplr(rgb_img)
                        thermal_img = np.fliplr(thermal_img)

                rgb_imgs.append(rgb_img)
                thermal_imgs.append(thermal_img)

            rgb_imgs = np.array(rgb_imgs)/127.5 - 1.
            thermal_imgs = np.array(thermal_imgs)[:,:,:,np.newaxis]/127.5 - 1.

            yield rgb_imgs, thermal_imgs


    def imread(self, path):
        try:
            img= cv2.imread(path)
            img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
        except:
            print(path)

    def thermal_imread(self,img_path):
        thermal_img_path= img_path
        thermal_img= skimage.io.imread(thermal_img_path)
        return thermal_img
