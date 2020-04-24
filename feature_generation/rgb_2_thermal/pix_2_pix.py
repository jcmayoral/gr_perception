import scipy
import cv2
from keras.datasets import mnist
from instance_normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
SEED=1234
np.random.seed(SEED)

class Pix2Pix():
    def __init__(self,img_rows=256,
                img_cols=256,
                channels=3,
                 thermal_channels=1,
                 dataset_name="flir_rgbdas",
                 max_batches = 20
                ):
        # Input shape
        self.max_batches = max_batches
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.thermal_channels=thermal_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.thermal_img_shape=(self.img_rows,self.img_cols,self.thermal_channels)

        # Configure data loader
        self.dataset_name = dataset_name

    def custom_initialize(self, rgb_dataset_folder, thermal_dataset_folder, path_timestamp_matching):
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols),
                                      rgb_dataset_folder=rgb_dataset_folder,
                                      thermal_dataset_folder=thermal_dataset_folder,
                                      path_timestamp_matching = path_timestamp_matching)


        # Calculate output shape of D (PatchGAN)
        patch = 64 #from the paper
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 128#64
        self.df = 128#64

        optimizer = Adam(0.0002,0.5,0.999)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_thermal = Input(shape=self.thermal_img_shape)
        img_rgb = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_thermal = self.generator(img_rgb)


        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_rgb,fake_thermal])

        self.combined = Model(inputs=[img_rgb, img_thermal], outputs=[valid, fake_thermal])
        self.combined.compile(loss=["mse","mae"],
                              loss_weights=[1,100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.thermal_channels, kernel_size=4, strides=1, padding='same', activation='tanh',name="thermal")(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True,strides=2):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_rgb = Input(shape=self.img_shape)
        img_thermal = Input(shape=self.thermal_img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_rgb, img_thermal])

        d1 = d_layer(combined_imgs, self.df, bn=False,strides=2) #128
        d2 = d_layer(d1, self.df*2,strides=2) #64
        #d3 = d_layer(d2, self.df*4,strides=1) #128
        #d4 = d_layer(d3, self.df*8,strides=1) #128
        #d5=  d_layer(d4, self.df*32,f_size=8)


        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',name="validity")(d2)

        return Model([img_rgb, img_thermal], validity)

    def train(self, epochs, batch_size=1, sample_interval=1):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_rgb, imgs_thermal) in enumerate(self.data_loader.load_batch(batch_size,thermal_ext=".jpeg")):
                if batch_i == self.max_batches:
                    print ("maximum number of batches per epoch is reached")
                    break
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Condition on B and generate a translated version
                fake_thermal = self.generator.predict(imgs_rgb)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_rgb, imgs_thermal], valid)
                d_loss_fake = self.discriminator.train_on_batch([imgs_rgb, fake_thermal], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_rgb, imgs_thermal], [valid, imgs_thermal])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch,batch_i,batch_size)

    def sample_images(self, epoch,batch_i, num_images=5):
        target_folder='current_results/{}/{}'.format(epoch,batch_i)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)#, exist_ok=True)
        r, c = num_images, 3

        imgs_rgb, imgs_thermal = self.data_loader.load_samples(num_images,thermal_ext=".tiff")
        fake_thermal = self.generator.predict(imgs_rgb)

        #TEST uncomment this if needed
        #np.save(target_folder+"/rgb.npy",imgs_rgb)
        #np.save(target_folder+"/original_thermal.npy",imgs_thermal)
        #np.save(target_folder+"/fake_thermal.npy",fake_thermal)
        imgs_thermal=0.5*imgs_thermal+0.5
        imgs_rgb=0.5*imgs_rgb+0.5
        fake_thermal=0.5*fake_thermal+0.5


        titles = ['Condition','Original', 'Generated']
        plt.figure(figsize=(5,5))
        fig, axs = plt.subplots(r, c,figsize=[20,20])
        cnt = 0
        for i in range(r):
            axs[i,0].imshow(imgs_rgb[i])
            axs[i,1].imshow(imgs_thermal[i][:,:,0],cmap="hot")
            axs[i,2].imshow(fake_thermal[i][:,:,0],cmap="hot")

            for j in range(c):
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
        fig.savefig("current_results/{}/{}/image.png".format(epoch,batch_i))


        plt.close()

        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")#, exist_ok=True)
        #print("uncomment save_weights after time matching has been implemented")
        #self.generator.save_weights("saved_models/{}_batch_{}.h5".format(epoch,batch_i))
        #instead store one
        self.generator.save_weights("saved_models/last_model.h5")
