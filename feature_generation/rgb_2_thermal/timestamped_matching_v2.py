import os
import decimal
import numpy as np
import rospy

class TimeStampedMatcher():
    def __init__(self, rgb_dataset_folder, thermal_dataset_folder):
        self.rgb_dataset_folder=rgb_dataset_folder
        self.thermal_dataset_folder=thermal_dataset_folder
        #TODO match images
        #Get them by order
        self.rgb_images_list=[image_name for image_name in sorted(os.listdir(self.rgb_dataset_folder)) if image_name.endswith("png")]
        self.thermal_images_list=[image_name for image_name in sorted(os.listdir(self.thermal_dataset_folder)) if image_name.endswith("jpg")]

    def execute(self):
        rootfilename = "_bag_"
        try:
            os.makedirs("depthfiles")
        except:
            pass
        os.chdir("depthfiles")

        print "storing rgb images timestamps by bag"

        rgb_list = list()

        for file in self.rgb_images_list:
            i = file.replace("image_filter_bag","")
            i = i.replace(".png","")
            bag_file = "rgb" + rootfilename+i[0]+".txt"
            #remove "_"
            i = i[10:-4]
            with open(bag_file, "a+") as f:
                f.write(i+"\n")
            rgb_list.append(i)

        print "storing thermal images timestamps by bag"

        thermal_list = list()

        for file in self.thermal_images_list:
            j = file.replace("image_filter_bag","")
            i = j.replace(".jpg","")
            bag_file = "thermal" + rootfilename+i[0]+".txt"
            #remove "_"
            i = i[10:-4]
            with open(bag_file, "a+") as f:
                f.write(i+"\n")
            thermal_list.append(i)

        print "creating matching files"
        os.chdir("..")
        try:
            os.makedirs("depthmatching")
        except:
            pass
        os.chdir("depthmatching")

        rgb_array = np.asarray(rgb_list, dtype=np.int64)
        thermal_array = np.asarray(thermal_list, dtype=np.int64)

        for file in self.rgb_images_list:
            i = file.replace("image_filter_bag","")
            i = i.replace(".png","")
            #remove "_"
            print i, "A"
            r = rospy.Time(int(i[:-4]))
            print r.to_nsec(),"aaaaaaaaaaaaaaaaaa"
            i = r.to_nsec()#i[10:-4]
            print i, "B"
            print np.abs(thermal_array - float(i)).min()
            filename = str(i)+".txt"
            timematching = thermal_array[np.abs(thermal_array - float(i)).argmin()]
            print timematching
            with open(filename, "w+") as f:
                f.write(str(timematching))

if __name__ == '__main__':
    ts = TimeStampedMatcher("/media/autolabel_traintest/train/openfield_all/0",
                            "/home/jose/ros_ws/src/gr_perception/feature_generation/bag_2_images/depth")
    ts.execute()
