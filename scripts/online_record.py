#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
from std_msgs.msg import Int64
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
        self.t_0 = time()
        self.Fs = 20.0  # Herz
        self.wd = 10.0  # seconds
        self.t_i = []
        self.data = OnlineProcess()
        self.shift = False
        self.update = 10
        self.hr = []
        self.hr_t = []
        self.hr_pub = rospy.Publisher("/time_serify/heart_rate", Int64, queue_size=100)

    # Callback to extract heart rate from data incoming from eigenvalues of head's motion
    def get_xyz_cb(self, msg):
        value = np.asarray([msg.x, msg.y, msg.z])
        t = time() - self.t_0
        if not self.shift:
            print('Calibrating...%f' % (round(self.update - t, 2)), end='\r')
        # Remove old  points if shifting is activated
        if self.shift:
            self.timeserie["time"] = self.timeserie["time"][1:]
            self.timeserie["values"] = self.timeserie["values"][1:, :]
        # Store  points
        self.timeserie["time"] = np.append(self.timeserie["time"], t)
        self.timeserie["values"] = np.vstack([self.timeserie["values"], value])

        # print(str(delta) + 'seconds elapsed')
        if t >= self.update:
            print('Update at %s seconds', t)
            # Start shifting
            self.shift = True
            self.update = t + 2

            # Interpolate at fixed frequency
            self.t_i = np.linspace(self.timeserie["time"][0], self.timeserie["time"][-1],
                                   self.timeserie["time"].size)
            interp_x = interp1d(self.timeserie["time"], self.timeserie["values"][:, 0])
            interp_x = self.butter_bandpass_filter(interp_x(self.t_i), 0.75, 4)
            interp_y = interp1d(self.timeserie["time"], self.timeserie["values"][:, 1])
            interp_y = self.butter_bandpass_filter(interp_y(self.t_i), 0.75, 4)
            interp_z = interp1d(self.timeserie["time"], self.timeserie["values"][:, 2])
            interp_z = self.butter_bandpass_filter(interp_z(self.t_i), 0.75, 4)

            # Keep only data for binary size until end
            self.data.largest_base = floor_log(len(self.t_i), 2)
            binary_range = self.t_i.size - self.data.largest_base
            self.data.t_i = self.t_i[binary_range:]
            # Load data in wavelt processing class
            self.data.interp_x = interp_x[binary_range:]
            self.data.interp_y = interp_y[binary_range:]
            self.data.interp_z = interp_z[binary_range:]
            # Execute wavelet processing on data
            hr = self.data.wvt_proc(show=False)
            self.hr.append(hr)
            self.hr_t.append(t)
            self.hr_pub.publish(hr)
            plt.figure(1)
            plt.plot(self.hr_t, self.hr)
            plt.xlabel('Time (s)')
            plt.ylabel('Heart Rate (BPM)')
            plt.title('Dynamic Heart Rate ')
            plt.pause(0.00001)

    def butter_bandpass(self, lowcut, highcut, order=5):
        nyq = 0.5 * self.Fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut):
        b, a = self.butter_bandpass(lowcut, highcut)
        y = filtfilt(b, a, data)
        return y


if __name__ == '__main__':
    counter = 0
    timeseries = SeriesConverter()
    plt.ion()
    plt.ion()
    plt.show()
    rospy.init_node("time_series_process")
    rospy.Subscriber("/centroid_XYZ", Point, timeseries.get_xyz_cb)
    rospy.spin()
