from keras.applications import MobileNet
from models.append_layers import extend_model2
import sys
import os
from models.unet import unet, sample_images2
import numpy as np

ROOT_PATH = "/media/WIP/rgb2thermal"

im_size = (128,128)

model = MobileNet(input_shape=(im_size[0], im_size[1],  3), include_top=False, alpha=0.25)
model = extend_model2(model,multiplier=2)
model.summary()

from keras.utils import plot_model
plot_model(model, to_file='extendedmodel2.png')

try:
    os.mkdir("testing3")
except:
    pass

os.chdir("testing3")

#TRAINING model
from keras.losses import binary_crossentropy, mean_absolute_error, categorical_crossentropy, sparse_categorical_crossentropy
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from generator import SuperGeneratorV2
import copy

model_id = "trashabletraining"
model_cp_cb = ModelCheckpoint(model_id+'.h5', save_best_only=True)
loss_mode = categorical_crossentropy

model.compile(optimizer=RMSprop(0.001), loss= loss_mode, # SumOfLosses(loss_mode, mean_absolute_error),
                metrics= ['accuracy'])#,Recall()])

#test
batch_size=20
n_epochs=50
#increase the size of the image (instead of reducing the crop we enlarge the "nibio")
traingenerator = SuperGeneratorV2(model=None,root_dir="/media/autolabel_traintest/train/",
                                batch_size=batch_size, use_perc=0.15, flip_images=True,
                                validation_split=0.1, test_split=0.01, image_size = (128,128),
                                filter_datasets=["openfield_all"], n_classes=4, add_noise=False, add_shift=True)
#hack to validation on generator
val_steps = 0#np.floor((traingenerator.trainsamples*0.15)/batch_size)
#two epochs to observe all data
steps = np.floor(traingenerator.trainsamples/batch_size) - val_steps
print ("training_steps %d validation_steps %d"%(steps, val_steps))
history = model.fit_generator(traingenerator.generator(),
                                validation_data=traingenerator.validation_data,
                               steps_per_epoch=steps,
                               epochs=n_epochs)
                               #callbacks=[model_cp_cb])#, EarlyStopping()])
traingenerator.stop_iterator()
#loss_id= ''.join(str(e) for e in traindataset)
#loss_id += model_id
#history_data[loss_id] = copy.deepcopy(history.history)
model_name ="mobilenet"
for i in range(10):
    sample_images2(model, traingenerator, "testing_sample_{}".format(str(i)) + model_name, num_images=5)
