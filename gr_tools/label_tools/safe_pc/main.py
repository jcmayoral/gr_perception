from safe_pc import PCProcessingClient
import rospy


if __name__ == "__main__":
    rospy.init_node('pc_fieldsafe_label')
    import os, sys
    try:
        os.mkdir("testdataset")
    except:
        pass
        #sys.exit()
    try:
        os.chdir("testdataset")
    except:
        sys.exit()


    import rosbag
    bag = rosbag.Bag(FILEPATH)
    pclabeler = PCProcessingClient()
    filename = ""
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        #rospy.sleep(0.05)
        filename = str(msg.header.stamp.to_nsec())
        current_result = pclabeler.call(msg)
        with open(filename,'a') as f:
            if len(current_result.found_objects.objects)> 0:
                print(current_result)

            for detection in current_result.found_objects.objects:
                if detection.pose.position.x > 0:
                    #x: -0.155390086843 FILTER the fucking car that it's detected all frmes
                    f.write("%f %f %f "%(detection.pose.position.x, detection.pose.position.y, detection.pose.position.z))
                    f.write("%f %f %f %f "%(detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z,  detection.pose.orientation.w))
                    f.write("%f %f %f "%(detection.speed.x, detection.speed.y, detection.speed.z))
                    f.write("%f \n"%(detection.is_dynamic))

    bag.close()
    #rospy.spin()
