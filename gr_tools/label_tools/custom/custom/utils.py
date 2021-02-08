import rosbag
import rospy
from sensor_msgs.msg import CameraInfo

def match_timestamps(rgb_stamps, depth_stamps):
    if len(rgb_stamps) < len(depth_stamps):
        return

def extract_camera_info(bagfile, info_topic):
    msg = None
    with rosbag.Bag(bagfile, 'r') as rbag:
        msgs = rbag.read_messages(info_topic)
        #print outbag.get_type_and_topic_info()
        for i, m in enumerate(msgs):
            #print m[0], type(m[1])
            msg = m[1]
    return msg
            #print i,type(CameraInfo(m[1]))

def extract_timestamps_frombag(rbag, info_topic):
    stamps = []
    seqs = []
    #with rosbag.Bag(bagfile, 'r') as rbag:
    msgs = rbag.read_messages(info_topic)
    #print outbag.get_type_and_topic_info()
    for i, m in enumerate(msgs):
        #print m[0], type(m[1])
        seqs.append(m[1].header.seq)
        stamps.append(m[1].header.stamp.to_nsec())
    return seqs, stamps

def extract_timestamps(bagfile, info_topic):
    stamps = []
    with rosbag.Bag(bagfile, 'r') as rbag:
        msgs = rbag.read_messages(info_topic)
        #print outbag.get_type_and_topic_info()
        for i, m in enumerate(msgs):
            #print m[0], type(m[1])
            stamps.append(m[1].header.stamp.to_nsec())
    return stamps
