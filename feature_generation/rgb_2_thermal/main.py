from data_loader import DataLoader
import matplotlib.pyplot as plt
from tools import visualize
from pix_2_pix import Pix2Pix

#MAIN TODO IMPORTANT
#Create a hashtable or other to match rgb and thermal
#Pre or processing step to match as good as possible both images
#NOTE Rotating and trim might be helpful
#TODO Change to pytorch

num_imgs = 10
im_size = (128,128)
dataset_name = "garbage"
#todo check why thermanl chanels are three
thermal_channels = 1
n_epochs = 5
max_batches = -1


print("Uncomment np.save on final approach to save batch")

#TODO add as arg
if False:
    data_loader = DataLoader(dataset_name, img_res=im_size,
                 rgb_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                 thermal_dataset_folder="/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
                 path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching")

    rgb_imgs, thermal_imgs = data_loader.load_samples(num_imgs=num_imgs,thermal_ext=".tiff")
    #imshow a sample of images
    visualize(rgb_imgs, thermal_imgs)
    exit(1)

model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name,
                thermal_channels=thermal_channels, max_batches = max_batches)

#FOR FIELDSAFE
#For some reasons it is not parametrized the rgb and thermal
#model.custom_initialize("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
        #                "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
        #                path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
        #                match_by_timestamps = True)

#FOR FLIR
#Images already matched by name
model.custom_initialize("/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                        "/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                        path_timestamp_matching="",
                        match_by_timestamps = False)

model.train(n_epochs, batch_size=num_imgs, sample_interval=50)
