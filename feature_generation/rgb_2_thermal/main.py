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
num_imgs = 20
im_size = (128,128)
dataset_name = "fieldsafe"
#dataset_name = "fieldsafe"
neurons_factor = 1
#todo check why thermanl chanels are three
thermal_channels = 1
n_epochs = 5
max_batches = -1
data_percentage = 100

#TODO add as arg

if len(sys.argv) < 4:
    print "used argv 1 is n_epochs and argv 2 neurons_factor argv3 data_percentage"
    sys.exit()

if len(sys.argv) >3:
    data_percentage = float(sys.argv[3])

if len(sys.argv) >2:
    neurons_factor = int(sys.argv[2])

if len(sys.argv) >1:
    n_epochs = int(sys.argv[1])

model_name = "_pix2pix_without_filter_16times_{}_inputchannels{}_datapercent{}".format(str(neurons_factor),
                                str(input_channels),str(data_percentage))

#FOR FLIR
#Images already matched by name

if dataset_name == "flir":
    thermal_extension = ".jpeg"
    model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name, channels =1,
                thermal_channels=thermal_channels, max_batches = max_batches, output_folder = "{}_{}".format(dataset_name, model_name),
                thermal_extension = thermal_extension)
    model.custom_initialize("/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/RGB",
                        "/media/datasets/flir/FLIR_FREE/FLIR_ADAS_1_3/train/thermal_8_bit",
                        path_timestamp_matching="",
                        match_by_timestamps = False,
                        factor=neurons_factor, thermal_threshold=200,
                        input_channels=input_channels, data_percentage=data_percentage)
else:
    #FOR FIELDSAFE
    #For some reasons it is not parametrized the rgb and thermal
    thermal_extension = ".tiff"
    model = Pix2Pix(img_rows=im_size[0], img_cols=im_size[1], dataset_name= dataset_name,
                    thermal_channels=thermal_channels, max_batches = max_batches, output_folder = "{}_{}".format(dataset_name, model_name),
                    thermal_extension = thermal_extension)
    model.custom_initialize("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                    "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw",
                    path_timestamp_matching="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/matching",
                    match_by_timestamps = True,
                    factor = neurons_factor, thermal_threshold = 50,
                    input_channels=input_channels, data_percentage=data_percentage)

model.train(n_epochs, batch_size=num_imgs, sample_interval=50)
