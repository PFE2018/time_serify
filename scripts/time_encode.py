#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import scipy.fftpack as fft
from time import time


class SeriesConverter(object):
    def __init__(self):
        self.image = None
        self.timeserie = dict({"time": np.array([]), "values": np.array([])})
        self.Fs = 30.0

    def get_values2d(self, msg):
        bridge = CvBridge()
        t = (msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        try:
            self.image = bridge.imgmsg_to_cv2(msg, "16UC1")
            value = self.image[230][250]
            self.timeserie["time"] = np.append(self.timeserie["time"], t)
            self.timeserie["values"] = np.append(self.timeserie["values"], value)
            # INSERT PROCESSING HERE ###############################################
            if len(self.timeserie["values"]) == 300:  # Windows of last 5 seconds
                self.do_fft()
                # Reset window's buffers
                self.timeserie["time"] = np.array([])
                self.timeserie["values"] = np.array([])
            ########################################################################
        except CvBridgeError or TypeError as e:
            print(e)

    def do_fft(self):
        Fs = self.Fs  # sampling rate
        Ts = 1.0 / Fs  # sampling interval

        n = len(self.timeserie["time"])  # length of the signal
        frq = np.linspace(0.0, Fs / 2.0, n / 2)  # one side frequency range

        Y = fft.fft(self.timeserie["values"]) / n  # fft computing and normalization
        Y = 2.0 / n * np.abs(Y[:n // 2])

        plt.figure(1)
        plt.subplot(211)
        plt.xlabel('Time (s)')
        plt.ylabel('Depth (mm)')
        plt.plot(self.timeserie["time"] - self.timeserie["time"][0], self.timeserie["values"])
        plt.subplot(212)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.plot(frq, Y)
        plt.pause(0.000001)


if __name__ == '__main__':
    counter = 0
    timeseries = SeriesConverter()
    plt.ion()
    plt.show()
    rospy.init_node("time_series_prcss")
    rospy.Subscriber("/kinect2/sd/image_depth", Image, timeseries.get_values2d)
    #rospy.Subscriber("/kinect2/sd/points", Image, timeseries.get_values3d)
    rospy.spin()
