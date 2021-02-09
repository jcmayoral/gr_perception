from sensor_msgs.msg import CameraInfo
from collections import OrderedDict

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
    rtimes = []
    stamps = []
    start_time = None
    rstart_time = None
    #with rosbag.Bag(bagfile, 'r') as rbag:
    msgs = rbag.read_messages(info_topic)
    #print outbag.get_type_and_topic_info()
    for i, m in enumerate(msgs):
        #print m[0], type(m[1])
        if start_time is None:
            start_time = m[1].header.stamp.to_sec()
            rstart_time = m[2]
        seqs.append(m[1].header.seq)
        stamps.append(m[1].header.stamp.to_sec()- start_time)
        rtimes.append(m[2] - rstart_time)
    return seqs, stamps, rtimes

def extract_timestamps(bagfile, info_topic):
    stamps = []
    start_time = None
    rstart_time = None

    with rosbag.Bag(bagfile, 'r') as rbag:
        msgs = rbag.read_messages(info_topic)
        #print outbag.get_type_and_topic_info()
        for i, m in enumerate(msgs):
            #print m[0], type(m[1])
            if start_time is None:
                start_time = m[1].header.stamp.to_sec
                rstart_time = m[2]
            stamps.append(m[1].header.stamp.to_sec()-start_time)
    return stamps

def save_matches(matches):
    with open("matches.txt", "w") as f:
        for m in matches:
            f.write("{} {} \n".format(m[0],m[1]))

def stamps_to_dict(data):
    mydict = dict()
    for i in data:
        seq, stamp, rt = i.rstrip().split(" ")
        mydict[int(seq)] = stamp
    return OrderedDict(sorted(mydict.items(), key=lambda t: t[0]))
