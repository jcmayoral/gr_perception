import matplotlib.pyplot as plt
from tools.tools import visualize
from models.pix_2_pix import Pix2Pix
import sys

#MAIN TODO IMPORTANT
#Create a hashtable or other to match rgb and thermal
#Pre or processing step to match as good as possible both images
#NOTE Rotating and trim might be helpful
#TODO Change to pytorch
input_channels = 3
num_imgs = 3
im_size = (128,128)
dataset_name = "flir"
#dataset_name = "fieldsafe"
neurons_factor = 1
#todo check why thermanl chanels are three
thermal_channels = 1
n_epochs = 5
max_batches = -1
data_percentage = 100

#TODO add as arg

if len(sys.argv) < 4:
    print ("used argv 1 is n_epochs and argv 2 neurons_factor argv3 data_percentage")
    sys.exit()

if len(sys.argv) >3:
    data_percentage = float(sys.argv[3])

if len(sys.argv) >2:
    neurons_factor = int(sys.argv[2])

if len(sys.argv) >1:
    n_epochs = int(sys.argv[1])


dataset_name = "flir_{}".format(data_percentage)
#dataset_name = "fieldsafe_{}".format(data_percentage)

network_name = "pix2pix_factor_{}".format(str(neurons_factor))
#model_name = "_pix2pix_without_filter_16times_{}_inputchannels{}_datapercent{}".format(str(neurons_factor),
#                                str(input_channels),str(data_percentage))
model_name = dataset_name + network_name
#FOR FLIR
#Images already matched by name

if "flir" in dataset_name:
    thermal_extension = ".jpeg"
    model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name, channels =3,
                thermal_channels=1, max_batches = max_batches, output_folder = model_name,
                thermal_extension = thermal_extension)
    model.custom_initialize("/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                        "/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                        path_timestamp_matching="",
                        match_by_timestamps = False,
                        factor=neurons_factor, thermal_threshold=200,
                        data_percentage=data_percentage)
else:
    #FOR FIELDSAFE
    #For some reasons it is not parametrized the rgb and thermal
    thermal_extension = ".tiff"
    model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name,
                    thermal_channels=thermal_channels, max_batches = max_batches, output_folder = model_name,
                    thermal_extension = thermal_extension)
    model.custom_initialize("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                    "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
                    path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
                    match_by_timestamps = True,
                    factor = neurons_factor, thermal_threshold = 50,
                    data_percentage=data_percentage)

model.train(n_epochs, batch_size=num_imgs, sample_interval=50)
