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


class SeriesConverter(object):
    def __init__(self):
        self.image = None
        self.timeserie = dict({"time": [], "values": []})
        self.Fs = 20.0  # Herz
        self.wd = 120.0  # seconds
        self.t_i = []

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

        # Store  points
        self.timeserie["time"].append(t)
        self.timeserie["values"].append(value)
        print(str(t - self.timeserie["time"][0]) + 'seconds elapsed')

        if t - self.timeserie["time"][0] > self.wd:
            # Transfer to numpy array
            self.timeserie["time"] = np.asarray(self.timeserie["time"])
            self.timeserie["time"] = self.timeserie["time"] - self.timeserie["time"][0]
            self.timeserie["values"] = np.asarray(self.timeserie["values"])

            # Interpolate at fixed frequency
            self.t_i = np.arange(0, self.wd, 1 / self.Fs)
            interp_x = interp1d(self.timeserie["time"], self.timeserie["values"][:, 0])
            interp_x = self.butter_bandpass_filter(interp_x(self.t_i), 0.75, 4)
            freq, fft_x = self.do_fft(interp_x)
            interp_y = interp1d(self.timeserie["time"], self.timeserie["values"][:, 1])
            interp_y = self.butter_bandpass_filter(interp_y(self.t_i), 0.75, 4)
            _, fft_y = self.do_fft(interp_y)
            interp_z = interp1d(self.timeserie["time"], self.timeserie["values"][:, 2])
            interp_z = self.butter_bandpass_filter(interp_z(self.t_i), 0.75, 4)
            _, fft_z = self.do_fft(interp_z)
            print('Enter filename...')
            name = input()
            pickle.dump((self.t_i, interp_x, interp_y, interp_z, freq, fft_x, fft_y, fft_z), open(name+'.p', 'wb'))

            # Plot real and interpolated signal
            plt.figure(1)
            plt.clf()
            plt.subplot(211)
            plt.xlabel('Time (s)')
            plt.ylabel('Motion x (m)')
            plt.plot(self.timeserie["time"], self.timeserie["values"][:, 0])
            plt.plot(self.t_i, interp_x, '-r')
            plt.subplot(212)
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude x')
            plt.plot(freq, fft_x)


            # float bpm=freq*60


            plt.figure(2)
            plt.clf()
            plt.subplot(211)
            plt.xlabel('Time (s)')
            plt.ylabel('Motion y (m)')
            plt.plot(self.timeserie["time"], self.timeserie["values"][:, 1])
            plt.plot(self.t_i, interp_y, '-r')
            plt.subplot(212)
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude y')
            plt.plot(freq, fft_y)

            plt.figure(3)
            plt.clf()
            plt.subplot(211)
            plt.xlabel('Time (s)')
            plt.ylabel('Motion z (m)')
            plt.plot(self.timeserie["time"], self.timeserie["values"][:, 2])
            plt.plot(self.t_i, interp_z, '-r')
            plt.subplot(212)
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude z')
            plt.plot(freq, fft_z)
            plt.pause(0.000001)
            self.timeserie = dict({"time": [], "values": []})




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
    timeseries = SeriesConverter()
    plt.ion()
    plt.show()
    rospy.init_node("time_series_prcss")
    # rospy.Subscriber("/kinect2/sd/image_depth", Image, timeseries.get_values2d_cb)
    rospy.Subscriber("/filtered_pcloud", PointCloud2, timeseries.get_xyz_cb)
    rospy.spin()
