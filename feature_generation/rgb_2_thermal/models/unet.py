#https://github.com/zhixuhao/unet/blob/master/model.py
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as keras


def unet(pretrained_weights = None,input_size = (256,256,1), neuron_factor=16, loss = 'mse', compile=True):
    inputs = Input(input_size)
    conv1 = Conv2D(neuron_factor*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(neuron_factor*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(neuron_factor*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(neuron_factor*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(neuron_factor*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(neuron_factor*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(neuron_factor*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(neuron_factor*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(neuron_factor*64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(neuron_factor*64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(neuron_factor*32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(neuron_factor*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(neuron_factor*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(neuron_factor*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(neuron_factor*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(neuron_factor*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(neuron_factor*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(neuron_factor*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(neuron_factor*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(neuron_factor*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(neuron_factor*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(neuron_factor*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    model = Model(input = inputs, output = conv10)

    if compile:
        model.compile(optimizer = Adam(lr = 1e-4), loss = loss , metrics = ['accuracy', 'mse'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

import matplotlib.pyplot as plt
def sample_images(model, data_loader, name, num_images=5,thermal_ext=".jpeg"):
    target_folder='{}'.format(name)
    r, c = num_images, 3

    imgs_rgb, imgs_thermal = data_loader.load_samples(num_images,thermal_ext=thermal_ext)
    fake_thermal = model.predict(imgs_rgb)

    #imgs_thermal=0.5*imgs_thermal+0.5
    #imgs_rgb=0.5*imgs_rgb+0.5
    #fake_thermal=0.5*fake_thermal+0.5


    titles = ['Condition','Original', 'Generated']
    plt.figure(figsize=(5,5))
    fig, axs = plt.subplots(r, c,figsize=[20,20])
    cnt = 0
    for i in range(r):
        axs[i,0].imshow(imgs_rgb[i][:,:,0])
        axs[i,1].imshow(imgs_thermal[i][:,:,0])
        axs[i,2].imshow(fake_thermal[i][:,:,0])

        #print np.unique(fake_thermal[i][:,:,0])
        #print np.unique(imgs_thermal[i][:,:,0])

        for j in range(c):
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
    fig.savefig("sample_{}.png".format(name))
    plt.close()


def sample_images(model, emodel, generator, name, num_images=5,thermal_ext=".jpeg"):
    print("called correct function ")
    target_folder='{}'.format(name)
    r, c = num_images, 2
    [imgs_rgb,fake_thermal], gt = generator.chunk_generator(num_images,generator.trainsamples)
    print(gt)
    #print(imgs_rgb[0].shape)
    #fake_thermal = model.predict(imgs_rgb)
    predictions = emodel.predict([imgs_rgb,fake_thermal])
    #imgs_thermal=0.5*imgs_thermal+0.5
    #imgs_rgb=0.5*imgs_rgb+0.5
    #fake_thermal=0.5*fake_thermal+0.5


    titles = ['Original', 'Generated']
    plt.figure(figsize=(5,5))
    fig, axs = plt.subplots(r, c,figsize=[20,20])
    cnt = 0
    for i in range(r):
        print("extended_results ,", predictions[i])
        prediction=np.argmax(predictions[i])
        label = np.argmax(gt[i])
        axs[i,0].imshow(imgs_rgb[i][:,:,0])
        axs[i,1].imshow(fake_thermal[i][:,:,0])


        axs[i, 0].set_title(titles[0]+ " GT " + str(label))
        axs[i,0].axis('off')
        axs[i, 1].set_title(titles[1]+ " Predicition " + str(prediction))
        axs[i,1].axis('off')

    fig.savefig("sample_{}.png".format(name))
    plt.close()



def sample_images2(model, generator, name, num_images=5):
    print("called correct2 function ")
    target_folder='{}'.format(name)
    r, c = num_images, 1

    imgs_rgb, gts = generator.chunk_generator(num_images,generator.trainsamples)
    predictions = model.predict(imgs_rgb)

    titles = ['RESULT ']
    plt.figure(figsize=(5,1))
    fig, axs = plt.subplots(r, c,figsize=[10,10])
    cnt = 0
    for i in range(r):
        gtclass=np.argmax(gts[i])
        predictedclass=np.argmax(predictions[i])
        print(imgs_rgb[i].shape)
        axs[i].imshow(imgs_rgb[i])

        axs[i].set_title(titles[0]+ " GT CLASS " + str(gtclass)+ " PREDICTED CLASS " + str(predictedclass))
        axs[i].axis('off')
    fig.savefig("sample_{}.png".format(name))
    plt.close()
