#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from matplotlib import pyplot as plt
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time

class PointsToMotion(object):
    def __init__(self):
        self.delay = [time.time()]
        self.counter = 0
        self.xyzPub = rospy.Publisher()

    def get_xyz_cb(self, msg):
        xyz = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            xyz.append(p)
        xyz = np.asarray(xyz)
        toc = time.time()
        self.delay.append(toc)
        self.counter += 1
        if self.counter == 100:
            self.delay = [j - i for i, j in zip(self.delay[:-1], self.delay[1:])]
        assert True

if __name__ == '__main__':
    counter = 0
    motion = PointsToMotion()
    plt.ion()
    plt.show()
    rospy.init_node("pcl_to_motion")
    rospy.Subscriber("/filtered_pcloud", PointCloud2, motion.get_xyz_cb)
    #rospy.Subscriber("/kinect2/sd/points", Image, timeseries.get_values3d)
    rospy.spin()