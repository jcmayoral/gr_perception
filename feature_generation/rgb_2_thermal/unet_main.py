from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from data_loader import DataLoader
from unet import unet
import pickle

batch_size = 5
im_size = (128,128)
dataset_name = "flir_8"
network_name = "unet"
n_epochs = 5
steps_per_epoch = 8300

model = unet(input_size=(im_size[0], im_size[1],  3))
model.summary()

data_loader = DataLoader(dataset_name=dataset_name,
                         img_res=(im_size[0], im_size[1]),
                         rgb_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                         thermal_dataset_folder="/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                         path_timestamp_matching = "",
                         match_by_timestamps = False)

steps_per_epoch = int(len(data_loader.rgb_images_list) / batch_size)
history = model.fit_generator(data_loader.iterator(batch_size, ".jpeg"), steps_per_epoch= steps_per_epoch, epochs=n_epochs)
pickle.dump(history, open(dataset_name + network_name + ".p", "wb" ))
