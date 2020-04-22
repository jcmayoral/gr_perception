from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
#from pix_2_pix import Pix2Pix

num_imgs = 5
data_loader = DataLoader("test", img_res=(128, 128),
             rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
             thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw")
rgb_imgs, thermal_imgs = data_loader.load_samples(num_imgs=num_imgs,thermal_ext=".tiff")

#TODO args
if True:
    visualize(rgb_imgs, thermal_imgs)

#model = Pix2Pix()
