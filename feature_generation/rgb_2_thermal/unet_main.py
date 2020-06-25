from models.data_loader import DataLoader
import matplotlib.pyplot as plt
from tools.tools import visualize
from models.unet import unet, sample_images
import pickle
import os
import sys

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

input_channels = 3

neurons_factor = 1
n_epochs = 3

if len(sys.argv) < 4:
    print ("used argv 1 is n_epochs and argv 2 neurons_factor argv3 data_percentage")
    sys.exit()

if len(sys.argv) >3:
    data_percentage = float(sys.argv[3])

if len(sys.argv) >2:
    neurons_factor = int(sys.argv[2])

if len(sys.argv) >1:
    n_epochs = int(sys.argv[1])

batch_size = 50
im_size = (128,128)
dataset_name = "flir_{}".format(data_percentage)
network_name = "unet_factor_{}".format(str(neurons_factor))

if not os.path.exists(dataset_name + network_name):
    os.makedirs(dataset_name + network_name)#, exist_ok=True)
os.chdir(dataset_name + network_name)


model = unet(input_size=(im_size[0], im_size[1],  input_channels), neuron_factor=neurons_factor, loss = 'mse')
model.summary()


if "fieldsafe" not in dataset_name:
    print ("FLIR")
    thermal_extension = ".jpeg"
    data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1],1),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False, input_channels=input_channels)
    val_data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1],1),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/val/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/val/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False, input_channels=input_channels)


else:
    thermal_extension = ".tiff"

    data_loader = DataLoader(dataset_name, img_res=im_size,
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             match_by_timestamps = True, thermal_threshold=245, input_channels=input_channels)

    val_data_loader = DataLoader(dataset_name, img_res=im_size,
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             match_by_timestamps = True, thermal_threshold=245, input_channels= input_channels)
steps_per_epoch = int(len(data_loader.rgb_images_list) / batch_size*(percent/100.))
val_steps_per_epoch = int(steps_per_epoch*0.2)#int(len(val_data_loader.rgb_images_list) / batch_size)
print ("Steps per epoch {} Total batches {} Epochs{}".format(steps_per_epoch, int(len(data_loader.rgb_images_list) / batch_size), n_epochs))

from keras.callbacks import EarlyStopping, ModelCheckpoint
model_cp_cb = ModelCheckpoint('weights.h5', save_best_only=True)

history = model.fit_generator(data_loader.generator(batch_size, thermal_extension), steps_per_epoch= steps_per_epoch, epochs=n_epochs,
                                validation_data=val_data_loader.generator(batch_size, thermal_extension), validation_steps=val_steps_per_epoch,
                                callbacks=[model_cp_cb])
pickle.dump(history, open(dataset_name + network_name + ".p", "wb" ))

model.save_weights(dataset_name+network_name+".h5")

sample_images(model, data_loader, name=dataset_name +"_"+ network_name, num_images=5, thermal_ext=thermal_extension)
