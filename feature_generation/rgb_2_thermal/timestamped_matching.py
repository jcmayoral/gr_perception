import os

class TimeStampedMatcher():
    def __init__(self, rgb_dataset_folder, thermal_dataset_folder):
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        #TODO match images
        #Get them by order
        self.rgb_images_list=[image_name for image_name in sorted(os.listdir(self.rgb_dataset_folder)) if image_name.endswith("jpg")]
        self.thermal_images_list=[image_name for image_name in sorted(os.listdir(self.thermal_dataset_folder)) if image_name.endswith("tiff")]

    def execute(self, mode):
        rootfilename = "_bag_"

        for file in self.rgb_images_list:
            i = file.replace("image_filter_bag","")
            i = i.replace(".jpg","")
            bag_file = "rgb" + rootfilename+i[0]+".txt"
            #remove "_"
            i = i[2:]
            print i, bag_file
            with open(bag_file, "a+") as f:
                f.write(i+"\n")

        for file in self.thermal_images_list:
            j = file.replace("image_filter_bag","")
            i = j.replace(".tiff","")
            bag_file = "thermal" + rootfilename+i[0]+".txt"
            #remove "_"
            i = i[2:]
            print i, bag_file
            with open(bag_file, "a+") as f:
                f.write(i+"\n")


if __name__ == '__main__':
    ts = TimeStampedMatcher("/media/datasets/thermal_fieldsafe/dataset/_Multisense_left_image_rect_color",
                            "/media/datasets/thermal_fieldsafe/dataset/_FlirA65_image_raw")
    ts.execute()
