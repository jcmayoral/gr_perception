#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Pix2Pix

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/generative/pix2pix"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/generative/pix2pix.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This notebook demonstrates image to image translation using conditional GAN's, as described in [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). Using this technique we can colorize black and white photos, convert google maps to google earth, etc. Here, we convert building facades to real buildings.
#
# In example, we will use the [CMP Facade Database](http://cmp.felk.cvut.cz/~tylecr1/facade/), helpfully provided by the [Center for Machine Perception](http://cmp.felk.cvut.cz/) at the [Czech Technical University in Prague](https://www.cvut.cz/). To keep our example short, we will use a preprocessed [copy](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/) of this dataset, created by the authors of the [paper](https://arxiv.org/abs/1611.07004) above.
#
# Each epoch takes around 15 seconds on a single V100 GPU.
#
# Below is the output generated after training the model for 200 epochs.
#
# ![sample output_1](https://www.tensorflow.org/images/gan/pix2pix_1.png)
# ![sample output_2](https://www.tensorflow.org/images/gan/pix2pix_2.png)

# ## Import TensorFlow and other libraries

# In[2]:
import os
import time

from matplotlib import pyplot as plt
from IPython import display
import tensorflow as tf


PATH = "/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB/"#os.path.join(os.path.dirname(path_to_zip), 'facades/')
THERMAL_PATH = "/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit/"#os.path.join(os.path.dirname(path_to_zip), 'facades/')


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


import cv2
import skimage

def thermal_imread(img_path, rotate=False):
    thermal_img_path= img_path
    thermal_img= skimage.io.imread(thermal_img_path)
    if rotate:
        thermal_img = self.rotate_image(thermal_img)
    #if thermal_img.dtype != np.uint8:
    #    thermal_img = self.convert(thermal_img, 0, 255, np.uint8)
    return thermal_img

def load(image_file, thermal_file):
    input_image= cv2.imread(image_file)
    #todo load both
    thermal_image = thermal_imread(thermal_file)
    return input_image, thermal_image

#inp, re = load(PATH+'FLIR_08844.jpg', THERMAL_PATH+'FLIR_08844.jpeg')
# casting to int for matplotlib to show the image
#plt.figure()
#plt.imshow(inp/255.0)
#plt.figure()
#plt.imshow(re/255.0)
#plt.show()

# In[8]:


def resize(input_image, real_image, height, width):
    input_image = cv2.resize(input_image, (height,width))
    #HACK
    real_image = cv2.resize(real_image, (height,width))
    #input_image = tf.cast(input_image, tf.float32)
    #real_image = tf.cast(real_image, tf.float32)
    #real_image = tf.expand_dims(real_image, 0)
    #real_image = real_image.reshape(w,h,1)
    #input_image = tf.image.resize(input_image, [height, width],
                    #            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #real_image = tf.image.resize(real_image, [height, width],
                    #           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #print input_image.shape, real_image.shape, "OOOOOo"
    return input_image, real_image


# In[10]:
# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


import glob
from random import randint
#HACK
def load_tuplesample():
    train_dataset = glob.glob(PATH+'*.jpg')
    thermal_dataset = glob.glob(THERMAL_PATH+'*.jpeg')
    max_index = min(len(train_dataset), len(thermal_dataset))
    index = str(randint(1,max_index)).zfill(5)

    try:
        inp, re = load(PATH+'FLIR_{}.jpg'.format(index), THERMAL_PATH+'FLIR_{}.jpeg'.format(index))
        inp, re = resize(inp,re, IMG_HEIGHT, IMG_WIDTH)
        inp, re = normalize(inp, re)
        return inp,np.expand_dims(re, axis=2)
    except:
        print "ERROR ", "IMAGE ", index
        return load_tuplesample()


#train_dataset = train_dataset.map(load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
#train_dataset = train_dataset.shuffle(BUFFER_SIZE)
#train_dataset = train_dataset.batch(BATCH_SIZE)

"""
# In[16]:


test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)
"""

# ## Build the Generator
#   * The architecture of generator is a modified U-Net.
#   * Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
#   * Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
#   * There are skip connections between the encoder and decoder (as in U-Net).
#

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Input, Conv2DTranspose, ReLU, Concatenate, Dropout

#in this case number of channels of the thermal image
OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
    #initializer = tf.random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                    kernel_initializer="random_normal", use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())
    return result

import numpy as np

#down_model = downsample(3, 4)

input = Input(shape=[IMG_WIDTH, IMG_HEIGHT,3])
#down_result = down_model(input)

def upsample(filters, size, apply_dropout=False):
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer="random_normal",
                                    use_bias=False))
    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())
    return result


# In[21]:
#up_model = upsample(3, 4)
#up_result = up_model(down_result)

from unet import unet

def Generator():
    #model = unet(input_size = (IMG_HEIGHT,IMG_WIDTH,3), neuron_factor=2, loss = 'mse', compile=False)
    #model.summary()
    #return model
    inputs = Input(shape=[IMG_HEIGHT,IMG_WIDTH,3], name="error_layer")
    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
    ]
    last = Conv2DTranspose(OUTPUT_CHANNELS, 4,
                         strides=2,
                         padding='same',
                         kernel_initializer="random_normal",
                         activation='tanh') # (bs, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])
    x = last(x)
    return Model(inputs=inputs, outputs=x)

from tensorflow.keras.utils import plot_model
generator = Generator()
plot_model(generator, show_shapes=True)#, dpi=64)

#print input.shape
#gen_output = generator(input)
#print gen_output.shape
#plt.imshow(np.asarray(gen_output[0],dtype=np.uint8))
#plt.show()

# * **Generator loss**
#   * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
#   * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
#   * This allows the generated image to become structurally similar to the target image.
#   * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).

# The training procedure for the generator is shown below:

# In[25]:


LAMBDA = 100#np.float64(100)


# In[26]:

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    #l1_loss = tf.reduce_mean(abs(target - gen_output))
    #gan_loss = tf.cast(gan_loss,"float64")
    l1_loss =tf.reduce_mean(tf.abs(target - gen_output))
    #l1_loss = tf.cast(l1_loss,"float32")

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# ![Generator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gen.png?raw=1)
#

# ## Build the Discriminator
#   * The Discriminator is a PatchGAN.
#   * Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
#   * The shape of the output after the last layer is (batch_size, 30, 30, 1)
#   * Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
#   * Discriminator receives 2 inputs.
#     * Input image and the target image, which it should classify as real.
#     * Input image and the generated image (output of generator), which it should classify as fake.
#     * We concatenate these 2 inputs together in the code (`tf.concat([inp, tar], axis=-1)`)

from tensorflow.keras.layers import ZeroPadding2D, concatenate

def Discriminator():
    #initializer = tf.random_normal_initializer(0., 0.02)
    inp = Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='dinput_image')
    tar = Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1], name='dtarget_image')
    x = Concatenate(axis=-1)([inp, tar]) # (bs, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    zero_pad1 = ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = Conv2D(512, 4, strides=1,
                kernel_initializer="random_normal",
                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    batchnorm1 = BatchNormalization()(conv)
    leaky_relu = LeakyReLU()(batchnorm1)
    zero_pad2 = ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    last = Conv2D(1, 4, strides=1,
                kernel_initializer="random_normal")(zero_pad2) # (bs, 30, 30, 1)
    return Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
#plot_model(discriminator, show_shapes=True, dpi=64)

#disc_out = discriminator([input, gen_output])
#print disc_out.shape
#plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
#plt.colorbar()

# **Discriminator loss**
#   * The discriminator loss function takes 2 inputs; **real images, generated images**
#   * real_loss is a sigmoid cross entropy loss of the **real images** and an **array of ones(since these are the real images)**
#   * generated_loss is a sigmoid cross entropy loss of the **generated images** and an **array of zeros(since these are the fake images)**
#   * Then the total_loss is the sum of real_loss and the generated_loss
#

# In[30]:
from keras.losses import BinaryCrossentropy
loss_object = BinaryCrossentropy(from_logits=True)


# In[31]:


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    #geneated_los = tf.cast(generator_loss, "float64")
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# The training procedure for the discriminator is shown below.
#
# To learn more about the architecture and the hyperparameters you can refer the [paper](https://arxiv.org/abs/1611.07004).

# ![Discriminator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/dis.png?raw=1)
#

# ## Define the Optimizers and Checkpoint-saver
#

# In[32]:

from tensorflow.keras.optimizers import Adam

generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

# In[33]:

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = ModelCheckpoint(filepath='test_weights.hdf5', verbose=1, save_best_only=True)

#checkpoint = ModelCheckpoint(generator_optimizer=generator_optimizer,
#                                 discriminator_optimizer=discriminator_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator)


# ## Generate Images
#
# Write a function to plot some images during training.
#
# * We pass images from the test dataset to the generator.
# * The generator will then translate the input image into the output.
# * Last step is to plot the predictions and **voila!**

# Note: The `training=True` is intentional here since
# we want the batch statistics while running the model
# on the test dataset. If we use training=False, we will get
# the accumulated statistics learned from the training dataset
# (which we don't want)

# In[34]:


def generate_images(model, test_input, tar):
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        w,h,c = test_input.shape
        test_input = test_input.reshape(1,w,h,c)
        w,h,c = tar.shape
        tar = tar.reshape(w,h)
        prediction = model(test_input, training=True)
        s,w,h,c  = prediction.shape
        prediction2 = tf.reshape(prediction,[w,h])
        prediction2 = prediction2.eval() #here is your image Tensor :)

        print "CORRECT ?", prediction2
        #prediction = tf.reshape(prediction[0],(w,h))
        #prediction = tf.make_ndarray( tf.compat.v1.make_tensor_proto(prediction[0]))
        plt.figure(figsize=(15,15))
        display_list = [test_input[0], tar, prediction2]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            #print display_list[i]
            print type(display_list[i]), display_list[i].dtype, title[i]

            # getting the pixel values between [0, 1] to plot it.
            #if i ==2:
            #    display_list[i] = display_list[0][:, :]
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
    plt.show()

# In[35]:
#TODO

def get_batch():
    rgb = list()
    real = list()
    for i in range(BATCH_SIZE):
        r,re = load_tuplesample()
        rgb.append(r)
        real.append(re)
    a,b =  np.asarray(rgb, dtype=np.float32), np.asarray(real, dtype=np.float32)
    print a.shape, b.shape, "CHECKIT OUT"
    return a,b


#for example_input, example_target in data:
#    generate_images(generator, example_input, example_target)


# ## Training
#
# * For each example input generate an output.
# * The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
# * Next, we calculate the generator and the discriminator loss.
# * Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
# * Then log the losses to TensorBoard.

# In[36]:

EPOCHS = 1


# In[37]:


import datetime
#log_dir="logs/"

#summary_writer = tf.summary.create_file_writer(
#  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


log_dir=""
summary_writer = tf.summary.FileWriter(
                log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    #discriminator.compile(loss='mse',
    #    optimizer="adam",
    #    metrics=['accuracy'])
    #discriminator.summary()
    valid = np.ones((BATCH_SIZE,30,30,1))
    fake = np.ones((BATCH_SIZE,30,30,1))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #gen_output = generator.predict(input_image)
        #disc_real_output = discriminator.train_on_batch([input_image, target],valid)
        #disc_generated_output = discriminator.train_on_batch([input_image, gen_output],fake)

        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        #disc_real_output = [np.float64(v) for v in disc_real_output]
        #disc_generated_output = [np.float64(v) for v in disc_generated_output]
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    print "TOTAL LOSS {}".format(gen_total_loss)
    print "GAN LOSS {}".format(gen_gan_loss)
    print "L1 LOSS {}".format(gen_l1_loss)
    print "DISC LOSS {}".format(disc_loss)

    #TODO MAKE THIS WORK
    #with summary_writer as sm:
    #    sm.add_summary(tf.summary.scalar('gen_total_loss', gen_total_loss))#, step=epoch))
    #    sm.add_summary(tf.summary.scalar('gen_gan_loss', gen_gan_loss))#, step=epoch))
    #    sm.add_summary(tf.summary.scalar('gen_l1_loss', gen_l1_loss))#, step=epoch))
    #    sm.add_summary(tf.summary.scalar('disc_loss', disc_loss))#, step=epoch))


# The actual training loop:
#
# * Iterates over the number of epochs.
# * On each epoch it clears the display, and runs `generate_images` to show it's progress.
# * On each epoch it iterates over the training dataset, printing a '.' for each example.
# * It saves a checkpoint every 20 epochs.

# In[39]:


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        #display.clear_output(wait=True)
        img_batch, real_batch = get_batch()

        #for (example_input, example_target) in zip(img_batch, real_batch):
        #    generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)
        # Train
        #for n, (input_image, target) in enumerate(zip(img_batch, real_batch)):
        #    print('.')
        #    if (n+1) % 100 == 0:
        #        print()
        #    train_step(input_image, target, epoch)
        #print()
        train_step(img_batch, real_batch, epoch)

        # saving (checkpoint) the model every 20 epochs
        #if (epoch + 1) % 20 == 0:
            #checkpoint.save(file_prefix = checkpoint_prefix)
        #print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
        #                                                time.time()-start))
        #checkpoint.save(file_prefix = checkpoint_prefix)


# This training loop saves logs you can easily view in TensorBoard to monitor the training progress. Working locally you would launch a separate tensorboard process. In a notebook, if you want to monitor with TensorBoard it's easiest to launch the viewer before starting the training.
#
# To launch the viewer paste the following into a code-cell:

# In[ ]:


#docs_infra: no_execute
#get_ipython().magic(u'load_ext tensorboard')
#get_ipython().magic(u'tensorboard --logdir {log_dir}')


# Now run the training loop:

# In[40]:

train_dataset = 1
test_dataset = 2
fit(train_dataset, EPOCHS, test_dataset)


# If you want to share the TensorBoard results _publicly_ you can upload the logs to [TensorBoard.dev](https://tensorboard.dev/) by copying the following into a code-cell.
#
# Note: This requires a Google account.
#
# ```
# !tensorboard dev upload --logdir  {log_dir}
# ```

# Caution: This command does not terminate. It's designed to continuously upload the results of long-running experiments. Once your data is uploaded you need to stop it using the "interrupt execution" option in your notebook tool.

# You can view the [results of a previous run](https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw) of this notebook on [TensorBoard.dev](https://tensorboard.dev/).
#
# TensorBoard.dev is a managed experience for hosting, tracking, and sharing ML experiments with everyone.
#
# It can also included inline using an `<iframe>`:

# In[41]:


#display.IFrame(
#    src="https://tensorboard.dev/experiment/lZ0C6FONROaUMfjYkVyJqw",
#    width="100%",
#    height="1000px")


# Interpreting the logs from a GAN is more subtle than a simple classification or regression model. Things to look for::
#
# * Check that neither model has "won". If either the `gen_gan_loss` or the `disc_loss` gets very low it's an indicator that this model is dominating the other, and you are not successfully training the combined model.
# * The value `log(2) = 0.69` is a good reference point for these losses, as it indicates a perplexity of 2: That the discriminator is on average equally uncertain about the two options.
# * For the `disc_loss` a value below `0.69` means the discriminator is doing better than random, on the combined set of real+generated images.
# * For the `gen_gan_loss` a value below `0.69` means the generator i doing better than random at foolding the descriminator.
# * As training progresses the `gen_l1_loss` should go down.

# ## Restore the latest checkpoint and test

# In[42]:


#get_ipython().system(u'ls {checkpoint_dir}')


# In[43]:


# restoring the latest checkpoint in checkpoint_dir
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ## Generate using test dataset

# In[44]:


# Run the trained model on a few examples from the test dataset
img_batch, real_batch = get_batch()

for inp, tar in zip(img_batch, real_batch):
  generate_images(generator, inp, tar)
