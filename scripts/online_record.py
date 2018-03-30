#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import scipy.fftpack as fft
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, filtfilt
from time import time
import pickle
from scripts.online_process import OnlineProcess, floor_log


class SeriesConverter(object):
    def __init__(self):
        self.image = None
        self.timeserie = dict({"time": np.empty([0, 1]), "values": np.empty([0, 3])})
        self.Fs = 20.0  # Herz
        self.wd = 10.0  # seconds
        self.t_i = []
        self.data = OnlineProcess()
        self.shift = False
        self.update = 20

    def get_values2d_cb(self, msg):
        bridge = CvBridge()
        t = (msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        try:
            self.image = bridge.imgmsg_to_cv2(msg, "16UC1")
            value = self.image[230][250]
            self.timeserie["time"].append(t)
            self.timeserie["values"].append(value)
            # INSERT PROCESSING HERE ###############################################
            if len(self.timeserie["values"]) == 300:  # Windows of last 5 seconds
                self.timeserie["time"] = np.asarray(self.timeserie["time"])
                self.timeserie["values"] = np.asarray(self.timeserie["values"])
                self.do_fft()
                # Reset window's buffers
                self.timeserie = dict({"time": [], "values": []})
            ########################################################################
        except CvBridgeError or TypeError as e:
            print(e)

    def get_xyz_cb(self, msg):
        xyz = []
        # Extract list of xyz coordinates of point cloud
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            xyz.append(p)
        value = np.mean(np.asarray(xyz), axis=0)
        t = (msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

        # Remove old  points if shifting is activated
        if self.shift:
            self.timeserie["time"] = self.timeserie["time"][1:]
            self.timeserie["values"] = self.timeserie["values"][1:,:]
        # Store  points
        self.timeserie["time"] = np.append(self.timeserie["time"], t)
        self.timeserie["values"] = np.vstack([self.timeserie["values"], value])
        delta = t - self.timeserie["time"][0]
        # print(str(delta) + 'seconds elapsed')
        if delta >= self.update:
            #Start shifting
            self.shift = True
            self.update = 2
            # Transfer to numpy array
            print len(self.timeserie["time"])

            # Interpolate at fixed frequency
            span = range(self.timeserie["time"])
            self.t_i = np.arange(self.timeserie["time"][0], self.timeserie["time"][-1], span/self.timeserie["time"].size)
            interp_x = interp1d(self.timeserie["time"], self.timeserie["values"][:, 0])
            interp_x = self.butter_bandpass_filter(interp_x(self.t_i), 0.75, 4)
            interp_y = interp1d(self.timeserie["time"], self.timeserie["values"][:, 1])
            interp_y = self.butter_bandpass_filter(interp_y(self.t_i), 0.75, 4)
            interp_z = interp1d(self.timeserie["time"], self.timeserie["values"][:, 2])
            interp_z = self.butter_bandpass_filter(interp_z(self.t_i), 0.75, 4)

            # Keep only data for binary size until end
            self.data.largest_base = floor_log(len(self.t_i), 2)
            binary_range = self.data.t_i.size - self.data.largest_base
            self.data.t_i = self.t_i[binary_range:]
            self.data.interp_x = interp_x[binary_range:]
            self.data.interp_y = interp_y[binary_range:]
            self.data.interp_z = interp_z[binary_range:]
            # self.data.show_data()
            self.data.wvt_proc(show=False)
            print(str(np.mean(self.data.hr_kinect)))


    def butter_bandpass(self, lowcut, highcut, order=5):
        nyq = 0.5 * self.Fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype = 'band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut):
        b, a = self.butter_bandpass(lowcut, highcut)
        y = filtfilt(b, a, data)
        return y

    def do_fft(self, values):
        Fs = self.Fs  # sampling rate
        Ts = 1.0 / Fs  # sampling interval

        n = len(self.t_i)  # length of the signal
        frq = np.linspace(0.0, Fs / 2.0, n / 2)  # one side frequency range

        Y = fft.fft(values) / n  # fft computing and normalization
        Y = 2.0 / n * np.abs(Y[:n // 2])

        return frq, Y



if __name__ == '__main__':
    counter = 0
    timeseries = SeriesConverter(online=True)
    plt.ion()
    plt.show()
    rospy.init_node("time_series_prcss")
    # rospy.Subscriber("/kinect2/sd/image_depth", Image, timeseries.get_values2d_cb)
    rospy.Subscriber("/filtered_pcloud", PointCloud2, timeseries.get_xyz_cb)
    rospy.spin()
