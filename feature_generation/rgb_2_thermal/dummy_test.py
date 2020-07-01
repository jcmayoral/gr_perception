#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

#from keras.models import load_model
from models.data_loader2 import DataLoader2
from models.append_layers import extend_with_depth
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


im_size = (256,256)
batch_size = 20

data_loader = DataLoader2(dataset_name="dummy",
                          img_res=im_size,
                          rgb_dataset_folder="/media/autolabel_traintest/train/openfield_all/",
                          thermal_dataset_folder="/home/jose/ros_ws/src/gr_perception/feature_generation/bag_2_images/depth",
                          path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/depthmatching",
                          match_by_timestamps = True,
                          thermal_threshold=100, data_percentage = 10, rgb_ext="png", thermal_ext="npz", batch_size=batch_size)
"""
(rgb, depth), labels = data_loader.load_samples()
plt.figure()
plt.imshow(rgb[0])
plt.figure()
plt.imshow(depth[0].reshape(im_size[0],im_size[1]))
plt.show()
print (labels, labels.shape)
"""
print(data_loader.validation_data[0][0].shape)
print(data_loader.validation_data[0][1].shape)
print(data_loader.validation_data[1].shape)
model = extend_with_depth(im_size)
model.summary()



from keras.utils import plot_model
plot_model(model, to_file='disponsabledepthmodel.png')

try:
    os.mkdir("depthtesting")
except:
    pass

os.chdir("depthtesting")

from keras.losses import binary_crossentropy, mean_absolute_error, categorical_crossentropy, sparse_categorical_crossentropy
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from generator import SuperGeneratorV2
import copy

#model_cp_cb = ModelCheckpoint(model_id+'.h5', save_best_only=True)
#loss_mode = categorical_crossentropy

model.compile(optimizer="adam", loss= categorical_crossentropy,
                metrics= ['accuracy'])#,Recall()])
n_epochs=25
val_steps = 0#np.floor((traingenerator.trainsamples*0.15)/batch_size)
print (data_loader.train_samples)
steps = np.floor(data_loader.train_samples/batch_size) - val_steps
print ("training_steps %d validation_steps %d"%(steps, val_steps))
history = model.fit_generator(data_loader.generator(),
                               validation_data=data_loader.validation_data,
                               steps_per_epoch=steps,
                               epochs=n_epochs)

import matplotlib.pyplot as plt
print(history.history)
plt.figure()
x = np.arange(n_epochs)
plt.plot(x, history.history["loss"], label="loss fuction")
plt.plot(x, history.history["val_loss"], label="validation loss")
plt.legend()
plt.savefig(model_id+"lossfunctions.png")


plt.figure()
print(x)
plt.plot(x, history.history["accuracy"], label="accuracy")
plt.plot(x, history.history["val_accuracy"], label="validation accuracy")
plt.legend()
plt.savefig(model_id+"accs.png")
