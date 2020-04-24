import os
import decimal
import numpy as np

class TimeStampedMatcher():
    def __init__(self, rgb_dataset_folder, thermal_dataset_folder):
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        #TODO match images
        #Get them by order
        self.rgb_images_list=[image_name for image_name in sorted(os.listdir(self.rgb_dataset_folder)) if image_name.endswith("jpg")]
        self.thermal_images_list=[image_name for image_name in sorted(os.listdir(self.thermal_dataset_folder)) if image_name.endswith("tiff")]

    def execute(self):
        rootfilename = "_bag_"
        os.makedirs("files")
        os.chdir("files")

        print "storing rgb images timestamps by bag"

        rgb_list = list()

        for file in self.rgb_images_list:
            i = file.replace("image_filter_bag","")
            i = i.replace(".jpg","")
            bag_file = "rgb" + rootfilename+i[0]+".txt"
            #remove "_"
            i = i[2:]
            with open(bag_file, "a+") as f:
                f.write(i+"\n")
            rgb_list.append(i)

        print "storing thermal images timestamps by bag"

        thermal_list = list()

        for file in self.thermal_images_list:
            j = file.replace("image_filter_bag","")
            i = j.replace(".tiff","")
            bag_file = "thermal" + rootfilename+i[0]+".txt"
            #remove "_"
            i = i[2:]
            with open(bag_file, "a+") as f:
                f.write(i+"\n")
            thermal_list.append(i)

        print "creating matching files"
        os.chdir("..")
        os.makedirs("matching")
        os.chdir("matching")

        rgb_array = np.asarray(rgb_list, dtype=np.int64)
        thermal_array = np.asarray(thermal_list, dtype=np.int64)

        for file in self.rgb_images_list:
            i = file.replace("image_filter_bag","")
            i = i.replace(".jpg","")
            #remove "_"
            i = i[2:]
            filename = i+".txt"
            timematching = thermal_array[np.abs(thermal_array - float(i)).argmin()]
            with open(filename, "w+") as f:
                f.write(str(timematching))

if __name__ == '__main__':
    ts = TimeStampedMatcher("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                            "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw")
    ts.execute()
