#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

#from keras.models import load_model
from models.data_loader import DataLoader
#from models.append_layers import extend_model
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


im_size = (256,256)
data_loader = DataLoader(dataset_name="dummy",
                          img_res=im_size,
                          rgb_dataset_folder="/media/autolabel_traintest/train/openfield_all/0",
                          thermal_dataset_folder="/home/jose/ros_ws/src/gr_perception/feature_generation/bag_2_images/depth",
                          path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/depthmatching",
                          match_by_timestamps = True,
                          thermal_threshold=100, data_percentage = 100, rgb_ext="png", thermal_ext="jpg")
rgb, depth = data_loader.load_samples(num_imgs=2)
print(rgb.shape, depth.shape)


plt.figure()
plt.imshow(rgb[0])
plt.figure()
plt.imshow(depth[0].reshape(im_size[0],im_size[1]))
plt.show()
