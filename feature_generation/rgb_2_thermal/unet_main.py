from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from data_loader import DataLoader
from unet import unet, sample_images
import pickle
import os
import sys

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

print " "

neuron_factor = 1
n_epochs = 3

print "used argv 1 is n_epochs and argv 2 neuron_factor"

if len(sys.argv) >2:
    neuron_factor = int(sys.argv[2])

if len(sys.argv) >1:
    n_epochs = int(sys.argv[1])


batch_size = 50
im_size = (128,128)
dataset_name = "fielsafe"
network_name = "unet_factor_{}_masked".format(str(neuron_factor))

if not os.path.exists(dataset_name + network_name):
    os.makedirs(dataset_name + network_name)#, exist_ok=True)
os.chdir(dataset_name + network_name)


model = unet(input_size=(im_size[0], im_size[1],  3), neuron_factor=neuron_factor, loss = 'binary_crossentropy')
model.summary()


if dataset_name != "fieldsafe":
    thermal_extension = ".jpeg"
    data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1]),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False)
    val_data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1]),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/val/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/val/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False)


else:
    thermal_extension = ".tiff"

    data_loader = DataLoader(dataset_name, img_res=im_size,
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             match_by_timestamps = True, thermal_threshold=245)

    val_data_loader = DataLoader(dataset_name, img_res=im_size,
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             match_by_timestamps = True, thermal_threshold=245)

steps_per_epoch = int(len(data_loader.rgb_images_list) / batch_size)
val_steps_per_epoch = int(steps_per_epoch*0.05)#int(len(val_data_loader.rgb_images_list) / batch_size)

print ("Steps per epoch {} Total batches {} Epochs{}".format(steps_per_epoch, int(len(data_loader.rgb_images_list) / batch_size), n_epochs))
history = model.fit_generator(data_loader.generator(batch_size, thermal_extension), steps_per_epoch= steps_per_epoch, epochs=n_epochs,
                                validation_data=val_data_loader.generator(batch_size, thermal_extension), validation_steps=val_steps_per_epoch)
pickle.dump(history, open(dataset_name + network_name + ".p", "wb" ))

model.save_weights(dataset_name+network_name+".h5")

sample_images(model, data_loader, dataset_name +"_"+ network_name, num_images=5, thermal_ext=thermal_extension)
