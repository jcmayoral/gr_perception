import pypcd
import os
import random
import rospy
import pypcl

from sensor_msgs.msg import PointCloud2
class PersonsPCDReader:
    def __init__(self):
        rospy.init_node("pcd_reader")
        self.pub = rospy.Publisher("fake_cloud", PointCloud2, queue_size=1)
        self.pcd_path = "/media/datasets/persons_pcd"
        self.pcd_file_list=[pcd for pcd in sorted(os.listdir(self.pcd_path)) if pcd.endswith("pcd")]
        self.read_random()

    def read_random(self):
        filename = random.sample(self.pcd_file_list, 1)[0]
        print (filename)
        print (os.path.join(self.pcd_path, filename))
        pc = pypcd.PointCloud.from_path(os.path.join(self.pcd_path, filename))
        # pc.pc_data has the data as a structured array
        # pc.fields, pc.count, etc have the metadata
        # center the x field
        #pc.pc_data['x'] -= pc.pc_data['x'].mean()
        """
        #THIS WILL BE HELPFUL
        pc = PointCloud.from_msg(msg)
        pc.save('foo.pcd', compression='binary_compressed')
        # maybe manipulate your pointcloud
        pc.pc_data['x'] *= -1
        outmsg = pc.to_msg()
        """
        outmsg = pc.to_msg()
        self.pub.publish(outmsg)
