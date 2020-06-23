from models.data_json_loader import DataJSONLoader
import matplotlib.pyplot as plt
from tools.tools import visualize
from models.unet import unet, sample_images
import pickle
import os
import sys
import json

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

print " "

neuron_factor = 1
n_epochs = 3
percent = 10

if len(sys.argv) < 4:
    print "used argv 1 is n_epochs and argv 2 data_percentage and argv3 neuron_factor"
    sys.exit()

if len(sys.argv) >3:
    neuron_factor = int(sys.argv[3])

if len(sys.argv) >2:
    percent = float(sys.argv[2])

if len(sys.argv) >1:
    n_epochs = int(sys.argv[1])


batch_size = 2
im_size = (512,512)
dataset_name = "fieldsafe_{}percentdata_".format(percent)
network_name = "unet_factor_{}_json_greyscalemasked".format(str(neuron_factor))

if not os.path.exists(dataset_name + network_name):
    os.makedirs(dataset_name + network_name)#, exist_ok=True)
os.chdir(dataset_name + network_name)


model = unet(input_size=(im_size[0], im_size[1],  1), neuron_factor=neuron_factor, loss = 'binary_crossentropy')
model.summary()





thermal_extension = ".jpeg"
data_loader = DataJSONLoader(dataset_name=dataset_name,
                     img_res=(im_size[0], im_size[1],1),
                     rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                     json_file="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_annotations.json")
val_data_loader = DataJSONLoader(dataset_name=dataset_name,
                     img_res=(im_size[0], im_size[1],1),
                     rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/val/RGB",
                     json_file="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_annotations.json")

next(data_loader.generator())

steps_per_epoch = 10#int(len(data_loader.rgb_images_list) / batch_size*(percent/100.))
val_steps_per_epoch = int(steps_per_epoch*0.2)#int(len(val_data_loader.rgb_images_list) / batch_size)
print ("Steps per epoch {} Total batches {} Epochs{}".format(steps_per_epoch, int(len(data_loader.rgb_images_list) / batch_size), n_epochs))
#history = model.fit_generator(data_loader.generator(batch_size, thermal_extension), steps_per_epoch= steps_per_epoch, epochs=n_epochs,
#                                validation_data=val_data_loader.generator(batch_size, thermal_extension), validation_steps=val_steps_per_epoch)
#pickle.dump(history, open(dataset_name + network_name + ".p", "wb" ))

#model.save_weights(dataset_name+network_name+".h5")

sample_images(model, data_loader, dataset_name +"_"+ network_name, num_images=5, thermal_ext=thermal_extension)
