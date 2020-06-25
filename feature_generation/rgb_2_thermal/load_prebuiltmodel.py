from keras.applications import MobileNet, InceptionV3
from models.append_layers import extend_model2
import sys
import os
from models.unet import unet, sample_images2
import numpy as np

ROOT_PATH = "/media/WIP/rgb2thermal"

im_size = (256,256)

model_id = "disponsable_{}".format("MobileNet")
#model = InceptionV3(input_shape=(im_size[0], im_size[1],  3), include_top=True, classes=4, weights=None)
model = MobileNet(input_shape=(im_size[0], im_size[1],  3), include_top=False, alpha=0.25)
# Freeze n number of layers from the last
#for layer in model.layers: layer.trainable = False
# Check the trainable status of the individual layers
#for layer in model.layers: print(layer, layer.trainable)
model = extend_model2(model,multiplier=4)
model.summary()

from keras.utils import plot_model
plot_model(model, to_file='extendedmodel2.png')


try:
    os.mkdir("testing"+model_id)
except:
    pass

os.chdir("testing"+model_id)

from keras.losses import binary_crossentropy, mean_absolute_error, categorical_crossentropy, sparse_categorical_crossentropy
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from generator import SuperGeneratorV2
import copy

model_cp_cb = ModelCheckpoint(model_id+'.h5', save_best_only=True)
loss_mode = categorical_crossentropy

model.compile(optimizer=RMSprop(0.001), loss= loss_mode, # SumOfLosses(loss_mode, mean_absolute_error),
                metrics= ['accuracy'])#,Recall()])

batch_size=20
n_epochs=25
traingenerator = SuperGeneratorV2(model=None,root_dir="/media/autolabel_traintest/train/",
                                batch_size=batch_size, use_perc=0.25, flip_images=True,
                                validation_split=0.2, test_split=0.01, image_size = im_size,
                                filter_datasets=["openfield_all"], n_classes=4, add_noise=False, add_shift=True)
val_steps = 0#np.floor((traingenerator.trainsamples*0.15)/batch_size)
steps = np.floor(traingenerator.trainsamples/batch_size) - val_steps
print ("training_steps %d validation_steps %d"%(steps, val_steps))
history = model.fit_generator(traingenerator.generator(),
                                validation_data=traingenerator.validation_data,
                               steps_per_epoch=steps,
                               epochs=n_epochs,
                               callbacks=[model_cp_cb])#, EarlyStopping()])
traingenerator.stop_iterator()
del traingenerator

testgenerator = SuperGeneratorV2(model=None,root_dir="/media/autolabel_traintest/test/",
                                batch_size=batch_size, use_perc=0.25, flip_images=False,
                                validation_split=0.2, test_split=0.01, image_size = im_size,
                                filter_datasets=["openfield_all"], n_classes=4, add_noise=False, add_shift=False)

for i in range(30):
    sample_images2(model, testgenerator, "{}_testing_sample_{}".format(str(model_id),str(i)), num_images=2)

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
