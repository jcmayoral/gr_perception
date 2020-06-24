import random as rnd
from glob import iglob, glob
import cv2
import numpy as np
import os
import copy
import random
#from keras.utils import to_categorical
from itertools import product
from skimage.util import random_noise


classes = ["3", "2", "1", "0"]

class SuperGeneratorV2:
    def __init__(self, model, root_dir="/media/labeled_datasets/", max_attempts=50, validation_split=0.1,
                 test_split = 0.2, use_perc = 1, add_noise = False, flip_images= False, add_shift=False,
                 image_size = (600,600), batch_size = 16, filter_datasets= None, n_classes=0):
        self.root_dir = root_dir
        self.model = model
        self.weights = dict()
        for i in classes:
            self.weights[i] = 1#/(1+int(i))#np.exp(-int(i))
        print (self.weights)

        self.add_shift = add_shift
        self.add_noise = add_noise
        self.flip_images = flip_images
        self.n_classes = n_classes
        self.set_class_dictionaries()


        self.queue_files = dict()
        self.all_files = dict()
        self.readsamples_counter = 0
        datasets_list = self.get_datasets_names()
        filtered_list = list()
        print (filter_datasets)

        if filter_datasets is not None:
            for f in filter_datasets:
                for d in datasets_list:
                    #print (d, f)
                    if f in d:
                        print ("adding dataset ", d , " with ", f)
                        filtered_list.append(d)
        else:
            self.datasets_list = dataset_list

        self.datasets_list = filtered_list

        print ("Reading ", self.datasets_list)

        self.classes = classes

        self.onehot_encoder = self.set_encoder()
        self.image_size = image_size
        print ("BAZ", batch_size)

        self.batch_size = batch_size
        self.max_attempts = max_attempts

        self.skip_files = None
        self.test_files = dict()

        self.find_all_files()
        self.data_samples = int(use_perc * self.get_total_samples(filter_datasets) )
        print("Data samples %d per classs %d" %(self.data_samples, self.data_samples/4))

        self.all_files = dict()
        self.find_all_files(True, self.data_samples/4)
        self.data_samples = self.get_total_samples(filter_datasets)
        print("FINAL data samples %d " %(self.data_samples))


        self.testingsamples = int(test_split * self.data_samples)
        self.validationsamples = int(validation_split * (self.data_samples - self.testingsamples))
        self.trainsamples = self.data_samples - self.validationsamples - self.testingsamples
        print ("datasets samples %d test %d validation %d train %d"% (self.data_samples, self.testingsamples,
                self.validationsamples, self.trainsamples))

        self.select_testvalidationset()
        #self.class_indexes = np.zeros(batch_size)
        #self.batch_filenames = list()
        #self.readsamples_counter = 0
        #print (len(self.all_files[self.datasets_list[1]]))

    def get_data_samples(self):
        return self.data_samples

    def set_encoder(self):
        if self.n_classes != 1:
            encoder = np.eye(self.n_classes, dtype=np.uint8)
            return encoder

        return [0.0,1.0]

    def set_class_dictionaries(self):
        #TODO for loop enumerate
        #class 0 -> Danger class 3 -> Safe
        if self.n_classes == 4:
            self.classes_dict = {"0": 0, "1": 1, "2": 2, "3": 3}
            self.in_classes_dict = {3:"Safe", 2: "Warn", 1:"Unsafe", 0:"Danger"}
            print (self.classes_dict, self.in_classes_dict, self.n_classes)
        #MEGA HACK
        if self.n_classes == 3:
            self.classes_dict = {"3": 2, "2": 1, "1": 1, "0": 0}
            self.in_classes_dict = {2:"Safe",1:"Unsafe",0:"Danger"}
            print (self.classes_dict)

        if self.n_classes == 2:
            self.classes_dict = {"3": 1, "2": 1, "1": 0, "0": 0}
            self.in_classes_dict = {1:"Safe",0:"Danger"}
            print (self.classes_dict)

        if self.n_classes == 1:
            self.classes_dict = {"3": 1, "2": 1, "1": 1, "0": 0}
            self.in_classes_dict = {1:"Safe", 0:"Danger"}
            print (self.classes_dict)


    def get_datasets_names(self):
        return [f for f in iglob(self.root_dir+'*', recursive=True) if os.path.isdir(f)]


    def select_testvalidationset(self):
        self.restart_generator()
        #TODO function load_validation_data load_testing_data
        #Not so sure if I should save test or reload it as a generator
        #print("select test")
        test_data = self.chunk_generator(self.testingsamples, self.data_samples)
        #print("select validation")
        self.validation_data = self.chunk_generator(self.validationsamples, self.data_samples, is_validation=True)

        #Files that are remaining are the new total
        self.all_files = copy.deepcopy(self.queue_files)
        self.restart_generator()

    def restart_generator(self):
        self.readsamples_counter = 0
        self.queue_files = copy.deepcopy(self.all_files)

    def find_all_files(self, filter=False, limit =100000):
        for dataset in self.datasets_list:
            for cl in self.classes:
                path = os.path.join(dataset, cl)

                for g in glob(path+"/*"):
                    image = g
                    #save just name of the file not entire path
                    image_name = image.replace(path+"/",'')
                    if dataset not  in self.all_files.keys():
                        self.all_files.update({dataset : dict()})
                    if cl not in self.all_files[dataset]:
                        self.all_files[dataset].update({cl : list()})

                    self.all_files[dataset][cl].append(image_name)

        if filter:
            for dataset in self.datasets_list:
                for cl in self.classes:
                    if len(self.all_files[dataset][cl]) > limit:
                        print ("Dataset %s class %s size of %d resizing to %d: "% (dataset, cl, len(self.all_files[dataset][cl]), limit))
                        self.all_files[dataset][cl] = random.sample(self.all_files[dataset][cl], int(limit))

    def get_encoder(self,ind):
        return self.onehot_encoder[ind]

    def shift(self,image):
        a,b,c = image.shape
        M = np.float32([[1,0,random.randint(-10,10)],[0,1,random.randint(-10,10)]])
        dst = cv2.warpAffine(image,M,(b,a))
        return dst.reshape(a,b,c)

    def chunk_generator(self,o_batch_size, samples, is_validation=False):

        #IF FLIP IMAGES... SIZE WILL BE MULTIPLIED BY 2
        if self.flip_images:
            batch_size = 2* o_batch_size
        else:
            batch_size = o_batch_size

        #TODO analyze if other image formats will be allowed
        channels=3
        images = np.zeros((batch_size, self.image_size[0],self.image_size[1], channels), dtype=np.uint8)
        labels = np.zeros((batch_size, self.n_classes))
        weights = np.zeros((batch_size, 1))

        self.class_indexes = np.zeros(batch_size)
        self.batch_filenames = list()

        #ITERATE ON ORIGINAL BATCH SIZE
        for i in range(o_batch_size):
            readingattempt = 0
            #print (self.readsamples_counter, self.data_samples)

            if self.readsamples_counter >= samples:
                #NOTE StopIteration crashes fit
                #raise StopIteration
                #instead of stopping restart
                #TODO restart after everybatch -> callback
                #print("if you see this on the middle of an epoch correct it ... samples", samples)
                self.restart_generator()
                readingattempt = 0

            n_items = 0
            while n_items < 1:
                dataset = rnd.choice(list(self.queue_files.keys()))
                cl = rnd.choice(list(self.queue_files[dataset].keys()))
                path = os.path.join(dataset, cl)
                n_items = len(self.queue_files[dataset][cl])
                if readingattempt > 2000:
                    print (readingattempt)
                readingattempt +=1
            image = rnd.choice(self.queue_files[dataset][cl])
            #print (image)
            #save just name of the file not entire path
            image_name = os.path.join(path, image)

            self.readsamples_counter+=1

            #update queue_files
            self.queue_files[dataset][cl].remove(image)

            #next(iglob(path+"/*"))
            self.batch_filenames.append(image_name)
            #print (image, image_name)
            im = cv2.imread(image_name, cv2.IMREAD_COLOR)
            im = cv2.resize(im,(self.image_size[1], self.image_size[0]))

            #TODO consider this
            #im = random_noise(im, mode='gaussian', mean=0.1, var=0.001)
            if self.add_noise and not is_validation:
                im = random_noise(im, mode='s&p', amount=0.05)
                im = np.array(255*im, dtype = 'uint8')

            if self.add_shift and not is_validation:
                im = self.shift(im)

            ind = self.classes_dict[cl]

            images[i] = im
            self.class_indexes[i] = ind
            labels[i] = self.get_encoder(ind)

            weights[i] = self.weights[cl]

            if self.flip_images:
                flip_im = cv2.flip(im, 1 )
                images[i+o_batch_size] = flip_im
                self.class_indexes[i+o_batch_size] = ind
                labels[i+o_batch_size] = self.onehot_encoder[ind]
        if self.model is not None:
            modeloutput = self.model.predict(images)
            return ([images,modeloutput],labels)#, weights.reshape(-1,))
        else:
            return (images, labels)

    def generator(self):
        self.run = True
        while self.run:#Should be iteration run per epoch Stop function on stop_iterator
            yield self.chunk_generator(self.batch_size, self.trainsamples)

    def stop_iterator(self):
        self.run = False
        self.generator()

    def get_total_samples(self,filter_datasets):
        count = 0
        for d in self.all_files.keys():
            for c in self.all_files[d].keys():
                #print (len(self.all_files[d][c]))
                count+=len(self.all_files[d][c])
            #print (count)
        return count
