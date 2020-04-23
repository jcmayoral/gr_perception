from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from pix_2_pix import Pix2Pix

#MAIN TODO IMPORTANT
#Create a hashtable or other to match rgb and thermal
#Pre or processing step to match as good as possible both images
#NOTE Rotating and trim might be helpful
#TODO Change to pytorch

num_imgs = 5
im_size = (128,128)
dataset_name = "garbage"
thermal_channels = 3

#TODO add as arg
if False:
    data_loader = DataLoader(dataset_name, img_res=im_size,
                 rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                 thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw")

    rgb_imgs, thermal_imgs = data_loader.load_samples(num_imgs=num_imgs,thermal_ext=".tiff")
    #imshow a sample of images
    visualize(rgb_imgs, thermal_imgs)

model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name,  thermal_channels=thermal_channels)
#For some reasons it is not parametrized the rgb and thermal
model.custom_initialize("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                        "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw")
model.train(3, batch_size=4, sample_interval=1)
