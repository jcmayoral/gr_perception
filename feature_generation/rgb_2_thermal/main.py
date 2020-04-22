from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from pix_2_pix import Pix2Pix

num_imgs = 4
data_loader = DataLoader("test", img_res=(128, 128),
             rgb_dataset_folder="/home/jose/Pictures",
             thermal_dataset_folder="/home/jose/Pictures")
rgb_imgs, thermal_imgs = data_loader.load_samples(num_imgs=num_imgs,thermal_ext=".jpg")

#TODO args
if False:
    visualize(rgb_imgs, thermal_imgs)

model = Pix2Pix()
