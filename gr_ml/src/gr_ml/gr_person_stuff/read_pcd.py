import pcl
import os

class PersonsPCDReader:
    def __init__(self):
        self.pcd_path = "/media/datasets/persons_pcd"
        self.pcd_file_list=[pcd_file for pcd in sorted(os.listdir(self.pcd_path)) if pcd.endswith("pcd")]
