from keras.models import load_model
from models.data_loader import DataLoader
import sys
import os
from models.unet import unet, sample_images
import numpy as np

ROOT_PATH = "/media/WIP/rgb2thermal"
dataset_name = "fieldsafe"

#TODO add as arg

if len(sys.argv) < 3:
    print "used argv 1 is neurons_factor argv2 input_channels argv3 data_percentage"
    sys.exit()

data_percentage = int(sys.argv[3])
input_channels = int(sys.argv[2])
neurons_factor = int(sys.argv[1])

dataset_name = "fieldsafe_{}percentdata_".format(data_percentage)
model_name = "unet_factor_{}_greyscalemasked_inputchannels{}".format(str(neurons_factor), input_channels)
model_name = dataset_name + model_name
 
file_name = os.path.join(ROOT_PATH, model_name, model_name+".h5")
print file_name


im_size = (128,128)

model = unet(input_size=(im_size[0], im_size[1],  input_channels), neuron_factor=neurons_factor, loss = 'mse', compile=False)
#model = unet
model.load_weights(file_name)

# store weights before loading pre-trained weights
print model.layers
preloaded_layers = model.layers
preloaded_weights = []

for pre in preloaded_layers:
    preloaded_weights.append(pre.get_weights())

# load pre-trained weights
model.load_weights(file_name, by_name=True)

# compare previews weights vs loaded weights
for layer, pre in zip(model.layers, preloaded_weights):
    weights = layer.get_weights()

    if weights:
        if np.array_equal(weights, pre):
            print('not loaded', layer.name)
        else:
            print('loaded', layer.name)

#model.summary()
#fieldsafe
thermal_extension = ".tiff"
#openfield
thermal_extension = ".jpg"

data_loader = DataLoader(dataset_name, img_res=im_size,
             #rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             rgb_dataset_folder="/media/subset_labeled_datasets/train/openfield/Danger",
             #thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
             thermal_dataset_folder="/media/subset_labeled_datasets/train/openfield/Danger",
             path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
             #match_by_timestamps = True, thermal_threshold=245, input_channels=input_channels)
             match_by_timestamps = False, thermal_threshold=245, input_channels=input_channels)

imgs, thermalimgs = data_loader.load_samples(thermal_ext=thermal_extension)

try:
    os.mkdir("testing")
except:
    pass

os.chdir("testing")

for i in range(10):
    sample_images(model, data_loader, "testing_sample_{}".format(str(i)) + model_name, num_images=5,thermal_ext=thermal_extension)
#def sample_images(model, data_loader, name, num_images=5,thermal_ext=".jpeg"):
