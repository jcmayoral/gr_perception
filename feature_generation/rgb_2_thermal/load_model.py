#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

from keras.models import load_model
from models.data_loader import DataLoader
from models.append_layers import extend_model
import sys
import os
from models.unet import unet, sample_images
import numpy as np

ROOT_PATH = "/media/WIP/rgb2thermal"
dataset_name = "fieldsafe"

#TODO add as arg
print(len(sys.argv))
if len(sys.argv) < 3:
    print ("used argv 1 is neurons_factor argv2 input_channels argv3 data_percentage")
    sys.exit()

data_percentage = int(sys.argv[3])
input_channels = int(sys.argv[2])
neurons_factor = int(sys.argv[1])

dataset_name = "fieldsafe_{}percentdata_".format(data_percentage)
model_name = "unet_factor_{}_greyscalemasked_inputchannels{}".format(str(neurons_factor), input_channels)
model_name = dataset_name + model_name

file_name = os.path.join(ROOT_PATH, model_name, model_name+".h5")
print (file_name)

im_size = (128,128)

model = unet(input_size=(im_size[0], im_size[1],  input_channels), neuron_factor=neurons_factor, loss = 'mse', compile=False)
#model = unet
model.load_weights(file_name)

# store weights before loading pre-trained weights
print (model.layers)
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


emodel = extend_model(model,multiplier=2)
model.summary()
emodel.summary()

from keras.utils import plot_model
plot_model(emodel, to_file='extendedmodel.png')

#fieldsafe
thermal_extension = ".tiff"
#openfield
thermal_extension = ".jpg"

try:
    os.mkdir("testing2")
except:
    pass

os.chdir("testing2")

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
traingenerator = SuperGeneratorV2(model=model,root_dir="/media/autolabel_traintest/train/",
                                batch_size=batch_size, use_perc=0.25, flip_images=True,
                                validation_split=0.1, test_split=0.01, image_size = (128,128),
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


testgenerator = SuperGeneratorV2(model=None,root_dir="/media/autolabel_traintest/test/",
                                batch_size=batch_size, use_perc=0.25, flip_images=False,
                                validation_split=0.2, test_split=0.01, image_size = (128,128),
                                filter_datasets=["openfield_all"], n_classes=4, add_noise=False, add_shift=False)

for i in range(30):
    sample_images(model, emodel,testgenerator, "testing_sample_{}".format(str(i)) + model_name, num_images=2,thermal_ext=thermal_extension)

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
