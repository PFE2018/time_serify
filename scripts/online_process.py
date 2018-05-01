#!/usr/bin/env python
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pywt
import math
import pandas as pd
import peakutils as pk


def floor_log(num, base):
    if num < 0:
        raise ValueError("Non-negative number only.")

    if num == 0:
        return 0

    return base ** int(math.log(num, base))

def get_euclidean_distance(x, y, z):
    e_dis = []
    for ix, iy, iz in zip(x, y, z):
        e_dis.append(np.sqrt(ix ** 2 + iy ** 2 + iz ** 2))
    return e_dis

class OnlineProcess(object):
    def __init__(self):
        self.largest_base = 0
        self.t_i = []
        self.interp_x = []
        self.interp_y = []
        self.interp_z = []
        self.signals = []
        self.ref = []
        self.ref_hr = []
        self.ref_time = []
        self.hr_kinect = []
        self.kinect_time = []

    # Wavelet processing and show #
    def wvt_proc(self, show=False):
        num_level = int(np.log2(self.largest_base))
        slct_lvl = 3

        euclidean_distance = get_euclidean_distance(self.interp_x, self.interp_y, self.interp_z)
        wlt = pywt.Wavelet('db6')
        new_sig = pywt.swt(euclidean_distance, wavelet=wlt, level=num_level)

        # INSERT FOR LOOP FOR PLOTS
        if show:
            plt.figure()
            for i in range(1, num_level + 1):
                plt.subplot(4, 3, i)
                plt.title('Wavelet coefficient' + str(i))
                plt.plot(self.t_i, new_sig[-i][0])
            plt.pause(0.00001)

        # Get peaks-locs, compute interval
        loc_idx = pk.indexes(new_sig[-slct_lvl][0], min_dist=6, thres=0.0)
        if show:
            # Plot slected wavelet coefficient peaks
            plt.subplot(4, 3, slct_lvl)
            plt.plot(self.t_i[loc_idx], new_sig[-slct_lvl][0][loc_idx])
            plt.pause(0.000001)
        peaks = {'Time': loc_idx * 1.0 / 20.0}
        peaks_df = pd.DataFrame(peaks)

        # Compute mean interval and get heart rate with sliding window
        hr = (60.0 / (peaks_df.diff().mean().values)).flatten()
        hr_time = peaks_df.values.flatten()

        if show:
            # Show analysis
            plt.figure()
            plt.plot(hr_time, hr)
            plt.legend(['Kinect measurement', 'ECG Ground truth'])
            plt.pause(0.000001)

        if np.isnan(hr):
            assert True
        return np.mean(hr)


if __name__ == '__main__':
    data = OnlineProcess()
    data.wvt_proc(show=False)

    # Get range fitting for kinect values
    kinect_hr_end = min(data.kinect_time[0][-1], data.kinect_time[1][-1], data.kinect_time[2][-1])
    data.ref_hr = data.ref_hr[data.ref_time <= kinect_hr_end]
    data.ref_time = data.ref_time[data.ref_time <= kinect_hr_end]

    # interpolate axis over ref with right range
    hr_x_f = interpolate.interp1d(data.kinect_time[0], data.hr_kinect[0], fill_value="extrapolate")
    hr_y_f = interpolate.interp1d(data.kinect_time[1], data.hr_kinect[1], fill_value="extrapolate")
    hr_z_f = interpolate.interp1d(data.kinect_time[2], data.hr_kinect[2], fill_value="extrapolate")

    hr_x = np.array(hr_x_f(data.ref_time))
    hr_y = np.array(hr_y_f(data.ref_time))
    hr_z = np.array(hr_z_f(data.ref_time))
    hr = np.array([hr_x, hr_y, hr_z])
    mean_hr = np.mean(hr, axis=0)


    # Plot results
    plt.figure()
    plt.plot(data.ref_time, data.ref_hr)
    plt.plot(data.ref_time, hr_x)
    plt.plot(data.ref_time, hr_y)
    plt.plot(data.ref_time, hr_z)
    plt.legend(['Ground  truth', 'HR x axis', 'HR y axis', 'HR z axis'])
    plt.pause(0.000001)
    plt.figure()
    plt.plot(data.ref_time, data.ref_hr)
    plt.plot(data.ref_time, mean_hr)
    plt.pause(0.000001)
    plt.figure()
    plt.plot(data.ref_time, abs(data.ref_hr-mean_hr))
    mean = np.mean(abs(data.ref_hr-mean_hr))
    dev = np.std(abs(data.ref_hr-mean_hr))
    plt.pause(0.000001)

    assert True
