#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

from keras.models import load_model
from models.data_loader import DataLoader
from models.append_layers import extend_model
import sys
import os
import numpy as np
from models.pix_2_pix import Pix2Pix
from models.unet import sample_images

ROOT_PATH = "/media/WIP/rgb2thermal"

#TODO add as arg
print(len(sys.argv))
if len(sys.argv) < 2:
    print ("used argv 1 is neurons_factor argv2 data_percentage")
    sys.exit()

input_channels = 3#int(sys.argv[2])
data_percentage = float(sys.argv[2])
neurons_factor = int(sys.argv[1])

dataset_name = "flir_{}".format(data_percentage)
model_name = "unet_factor_{}".format(str(neurons_factor))
model_name = "pix2pix_factor_{}".format(str(neurons_factor))

model_name = dataset_name + model_name

#file_name = os.path.join(ROOT_PATH, model_name, model_name+".h5")
file_name = os.path.join(model_name,"saved_models", "modelflir_15.0.h5")

im_size = (128,128)
thermal_extension = ".jpeg"

#TODO trainable initilizaer
model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name, channels =3,
            thermal_channels=1, max_batches = 1, output_folder = model_name,
            thermal_extension = thermal_extension)
model.custom_initialize("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
            "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
            path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
            match_by_timestamps = True,
            factor = neurons_factor, thermal_threshold = 50,
            data_percentage=data_percentage)
model.generator.load_weights(file_name)

emodel = extend_model(model.generator,multiplier=4)
print(emodel.trainable)
emodel.summary()

from keras.utils import plot_model
plot_model(emodel, to_file='extendedmodel.png')

#fieldsafe
thermal_extension = ".tiff"
#openfield
thermal_extension = ".jpg"

try:
    os.mkdir("testingdisponsable_"+model_name)
except:
    pass

os.chdir("testingdisponsable_"+model_name)

#TRAINING emodel
from keras.losses import binary_crossentropy, mean_absolute_error, categorical_crossentropy, sparse_categorical_crossentropy
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from generator import SuperGeneratorV2
import copy

model_id = "disponsable_training"
model_cp_cb = ModelCheckpoint(model_id+'.h5', save_best_only=True)
loss_mode = categorical_crossentropy

emodel.compile(optimizer=RMSprop(0.001), loss= loss_mode, # SumOfLosses(loss_mode, mean_absolute_error),
                metrics= ['accuracy'])#,Recall()])

#test
batch_size=20
n_epochs=25
#increase the size of the image (instead of reducing the crop we enlarge the "nibio")
traingenerator = SuperGeneratorV2(model=model.generator,root_dir="/media/autolabel_traintest/train/",
                                batch_size=batch_size, use_perc=0.2, flip_images=True,
                                validation_split=0.15, test_split=0.01, image_size = (128,128),
                                filter_datasets=["openfield_all"], n_classes=4, add_noise=False, add_shift=True)
#hack to validation on generator
val_steps = 0#np.floor((traingenerator.trainsamples*0.15)/batch_size)
#two epochs to observe all data
steps = np.floor(traingenerator.trainsamples/batch_size) - val_steps
print ("training_steps %d validation_steps %d"%(steps, val_steps))
history = emodel.fit_generator(traingenerator.generator(),
                                validation_data=traingenerator.validation_data,
                               steps_per_epoch=steps,
                               epochs=n_epochs, workers=0,
                               callbacks=[model_cp_cb])#, EarlyStopping()])
traingenerator.stop_iterator()
del traingenerator


testgenerator = SuperGeneratorV2(model=model.generator,root_dir="/media/autolabel_traintest/test/",
                                batch_size=batch_size, use_perc=0.25, flip_images=False,
                                validation_split=0.2, test_split=0.01, image_size = (128,128),
                                filter_datasets=["openfield_all"], n_classes=4, add_noise=False, add_shift=False)

for i in range(30):
    sample_images(model.generator, emodel,testgenerator, "testing_sample_{}".format(str(i)) + model_name, num_images=2,thermal_ext=thermal_extension)

import matplotlib.pyplot as plt
print(history.history)
plt.figure()
x = np.arange(n_epochs)
plt.plot(x, history.history["loss"], label="loss fuction")
plt.plot(x, history.history["val_loss"], label="validation loss")
plt.legend()
plt.savefig("lossfunctions.png")


plt.figure()
print(x)
plt.plot(x, history.history["accuracy"], label="accuracy")
plt.plot(x, history.history["val_accuracy"], label="validation accuracy")
plt.legend()
plt.savefig("accs.png")
