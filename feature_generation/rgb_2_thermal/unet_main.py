from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from data_loader import DataLoader
from unet import unet, sample_images
import pickle
import os

batch_size = 15
im_size = (128,128)
dataset_name = "fielsafe"
network_name = "unet_4_masked"
n_epochs = 5
steps_per_epoch = 8300

if not os.path.exists(dataset_name + network_name):
    os.makedirs(dataset_name + network_name)#, exist_ok=True)
os.chdir(dataset_name + network_name)


model = unet(input_size=(im_size[0], im_size[1],  3), neurons_number=4, loss = 'binary_crossentropy')
model.summary()
"""
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
"""

thermal_extension = ".tiff"

data_loader = DataLoader(dataset_name, img_res=im_size,
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             match_by_timestamps = True)



val_data_loader = DataLoader(dataset_name, img_res=im_size,
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             match_by_timestamps = True)

steps_per_epoch = int(0.0025*len(data_loader.rgb_images_list) / batch_size)
val_steps_per_epoch = 10#int(len(val_data_loader.rgb_images_list) / batch_size)

print ("Steps per epoch {} Total batches {} Epochs{}".format(steps_per_epoch, int(len(data_loader.rgb_images_list) / batch_size), n_epochs))
history = model.fit_generator(data_loader.generator(batch_size, thermal_extension), steps_per_epoch= steps_per_epoch, epochs=n_epochs,
                                validation_data=val_data_loader.generator(batch_size, thermal_extension), validation_steps=val_steps_per_epoch)
pickle.dump(history, open(dataset_name + network_name + ".p", "wb" ))

model.save_weights(dataset_name+network_name+".h5")

sample_images(model, data_loader, dataset_name +"_"+ network_name, num_images=batch_size, thermal_ext=thermal_extension)
