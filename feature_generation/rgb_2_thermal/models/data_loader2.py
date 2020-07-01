import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform
import os
import cv2
import random
from sklearn import preprocessing
from filters.preprocessing import filter_thermal
import random as rnd
import copy

classes = ["3", "2", "1", "0"]

class DataLoader2():
    def __init__(self, dataset_name, img_res=(128, 128),
                 rgb_dataset_folder="/floyd/input/flir_adas/train/RGB",
                 thermal_dataset_folder="/floyd/input/flir_adas/train/thermal_8_bit",
                 path_timestamp_matching="~/", match_by_timestamps=False, thermal_threshold=127, data_percentage=100,
                 input_channels=3, rgb_ext="jpg", thermal_ext="tiff", batch_size=10):
        self.obatch_size = batch_size
        self.input_channels = input_channels
        self.dataset_name = dataset_name
        self.data_percentage = data_percentage/100.
        print ("original batch size ", self.obatch_size)
        print ("data percentage ", self.data_percentage)
        self.img_res = img_res
        self.thermal_res = (img_res[0], img_res[1],1)
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        self.path_timestamp_matching = path_timestamp_matching
        self.match_by_timestamps = match_by_timestamps
        self.thermal_threshold = thermal_threshold
        self.rgb_ext = rgb_ext
        self.thermal_ext = thermal_ext
        self.n_classes = len(classes)
        self.classes_dict = {"0": 0, "1": 1, "2": 2, "3": 3}
        self.flip_images = True

        if match_by_timestamps:
            self.match_thermalfunction= self.match_by_timestamp
        else:
            self.match_thermalfunction = self.match_by_name

        #TODO match images
        self.thermal_min_max_scaler = preprocessing.MinMaxScaler()
        self.rgb_images_list=[image_name for image_name in os.listdir(self.rgb_dataset_folder) if image_name.endswith(self.rgb_ext)]
        self.thermal_images_list=[image_name for image_name in os.listdir(self.thermal_dataset_folder) if image_name.endswith(self.thermal_ext)]

        self.classes = classes
        self.all_files = dict()
        self.find_all_files()
        self.queue_files = dict()


        total_samples = int(self.get_total_samples()*(data_percentage/100))
        self.validationsamples = int(total_samples*0.1)
        self.train_samples = total_samples - self.validationsamples
        print ("DATA percentage", data_percentage)
        print ("total samples ", total_samples)
        print ("training samples ", self.train_samples)
        print ("validation samples ", self.validationsamples)

        self.select_testvalidationset()

    def select_testvalidationset(self):
        self.restart_generator()
        #TODO function load_validation_data load_testing_data
        #test_data = self.load_samples(self.testingsamples, self.data_samples)
        self.validation_data = self.load_samples(rbatching= self.validationsamples)

        #Files that are remaining are the new total
        self.all_files = copy.deepcopy(self.queue_files)
        self.restart_generator()

    def restart_generator(self):
        self.queue_files = self.all_files
        self.readsamples_counter = 0

    def generator(self):
        self.run = True
        while self.run:#Should be iteration run per epoch Stop function on stop_iterator
            yield self.load_samples()

    def load_samples(self,thermal_ext=None,rbatching=None):
        print (rbatching)
        if rbatching is None:
            print ("AA")
            rbatching = self.o_batch_size

        if thermal_ext is None:
            thermal_ext="."+self.thermal_ext

        if self.flip_images:
            batch_size = 2* rbatching
        else:
            batch_size = rbatching

        rgb_imgs,thermal_imgs=[],[]
        #random_rgb_image_name_list=np.random.choice(self.rgb_images_list,size=num_imgs).tolist()
        channels=3
        images = np.zeros((batch_size, self.img_res[0],self.img_res[1], channels), dtype=np.uint8)
        depth = np.zeros((batch_size, self.img_res[0],self.img_res[1], 1), dtype=np.uint8)
        labels = np.zeros((batch_size, self.n_classes))
        image_list = list()

        for i in range(rbatching):
            if self.readsamples_counter >= self.train_samples:
                self.restart_generator()

            n_items = 0
            readingattempt = 0

            while n_items < 1:
                dataset = rnd.choice(list(self.queue_files.keys()))
                cl = rnd.choice(list(self.queue_files[dataset].keys()))
                path = os.path.join(dataset, cl)
                n_items = len(self.queue_files[dataset][cl])
                if readingattempt > 200:
                    print (readingattempt)
                    readingattempt +=1
                    self.restart_generator()

            image = rnd.choice(self.queue_files[dataset][cl])
            image_list.append(image)
            image_name = os.path.join(path, image)
            self.queue_files[dataset][cl].remove(image)
            im = cv2.imread(image_name, cv2.IMREAD_COLOR)
            im = cv2.resize(im,(self.img_res[1], self.img_res[0]))
            ind = self.classes_dict[cl]

            self.readsamples_counter+=1

            images[i] = im
            labels[i] = np.eye(4)[ind]

            if self.flip_images:
                flip_im = cv2.flip(im, 1 )
                images[i+rbatching] = flip_im
                labels[i+rbatching] =  np.eye(4)[ind]

        self.class_indexes = np.zeros(batch_size)
        self.batch_filenames = list()

        random_thermal_image_name_list = self.match_thermalfunction(image_list)

        for i,thermal_img_name in enumerate(random_thermal_image_name_list):
            thermal_img_path=os.path.join(self.thermal_dataset_folder,thermal_img_name.split(".")[0]+thermal_ext)
            thermal_img = self.thermal_imread(thermal_img_path)
            thermal_img = cv2.resize(thermal_img, (self.img_res[0], self.img_res[1]))
            thermal_img = thermal_img[:,:,np.newaxis]
            depth[i]= thermal_img

        return ([images, depth], labels)

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
            r = r.replace("."+self.rgb_ext,"")
            r = r[:-4]
            filename = os.path.join(self.path_timestamp_matching, r +".txt")
            with open(filename) as f:
                 thermal_index = f.read()
            #TODO this is totally bruteforce
            #if not found set a default
            needle_file = self.thermal_images_list[-1]
            for image_file in self.thermal_images_list:
                if thermal_index in image_file:
                    needle_file = image_file
                    break;
            thermal_files.append(needle_file)
        return np.asarray(thermal_files)

    #fieldsafe specific
    def rotate_image(self, image):
        (rows,cols) = image.shape
        #M = cv2.getRotationMatrix2D((rows/2, cols/2), 180, 1)
        #thermal_img = imutils.rotate(thermal_img, 180, dtype = np.float32)
        return transform.rotate(image,180)#cv2.warpAffine(image, M, (rows, cols))

    #fieldsafe specific
    def crop_image(self, image):
        return image[:,300:,:]

    def imread(self, path):
        #try:
        img= cv2.imread(path)
        #For FIELDSAFE
        #if self.match_by_timestamps:
        #    img = self.crop_image(img)
            #img = self.rotate_image(img)

        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.input_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def thermal_imread(self,img_path):
        thermal_img_path= img_path
        #thermal_img= io.imread(thermal_img_path)
        thermal_imgread = np.load(thermal_img_path)
        thermal_img = thermal_imgread['image']
        #if self.match_by_timestamps:
        thermal_img = self.rotate_image(thermal_img)
        thermal_img = thermal_img[:, ::-1]

        #if thermal_img.dtype != np.uint8:
        #    thermal_img = self.convert(thermal_img, 0, 255, np.uint8)
        return thermal_img

    def get_total_samples(self):
        count = 0
        for d in self.all_files.keys():
            for c in self.all_files[d].keys():
                #print (len(self.all_files[d][c]))
                count+=len(self.all_files[d][c])
            #print (count)
        return count

    def find_all_files(self, filter=False, limit =100000):
        for cl in self.classes:
            path = os.path.join(self.rgb_dataset_folder, cl)
            print (path)
            for g in glob(path+"/*"):
                image = g
                #save just name of the file not entire path
                image_name = image.replace(path+"/",'')
                if self.rgb_dataset_folder not  in self.all_files.keys():
                    self.all_files.update({self.rgb_dataset_folder : dict()})
                if cl not in self.all_files[self.rgb_dataset_folder]:
                    self.all_files[self.rgb_dataset_folder].update({cl : list()})

                self.all_files[self.rgb_dataset_folder][cl].append(image_name)

        if filter:
            for cl in self.classes:
                if len(self.all_files[self.rgb_dataset_folder][cl]) > limit:
                    print ("Dataset %s class %s size of %d resizing to %d: "% (self.rgb_dataset_folder, cl, len(self.all_files[self.rgb_dataset_folder][cl]), limit))
                    self.all_files[self.rgb_dataset_folder][cl] = random.sample(self.all_files[self.rgb_dataset_folder][cl], int(limit))
