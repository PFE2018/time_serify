import pickle
import matplotlib.pyplot as plt
from scipy import signal, interpolate, io
import numpy as np
import pywt
import math
import pandas as pd
import biosppy as bp


def floor_log(num, base):
    if num < 0:
        raise ValueError("Non-negative number only.")

    if num == 0:
        return 0

    return base ** int(math.log(num, base))


class OnlineProcess(object):
    def __init__(self):
        self.largest_base = 0
        self.t_i = []
        self.interp_x = []
        self.interp_y = []
        self.interp_z = []
        self.freq = []
        self.fft_x = []
        self.fft_y = []
        self.fft_z = []
        self.ref = []
        self.ref_hr = []
        self.ref_time = []
        self.hr_kinect = []
        self.kinect_time = []

    # Wavelet processing and show #
    def wvt_proc(self, show=True):
        num_level = int(np.log2(self.largest_base))
        slct_lvl = 4
        for axis in [self.interp_x, self.interp_y, self.interp_z]:
            wlt = pywt.Wavelet('db6')
            new_sig = pywt.swt(axis, wavelet=wlt, level=num_level)

            if show:
                plt.figure()
                plt.subplot(421)
                plt.title('Wavelet coefficient 1')
                plt.plot(self.t_i, new_sig[6][1])
                plt.subplot(422)
                plt.title('Wavelet coefficient 2')
                plt.plot(self.t_i, new_sig[5][1])
                plt.subplot(423)
                plt.title('Wavelet coefficient 3')
                plt.plot(self.t_i, new_sig[4][1])
                plt.subplot(424)
                plt.title('Wavelet coefficient 4')
                plt.plot(self.t_i, new_sig[3][1])
                plt.subplot(425)
                plt.title('Wavelet coefficient 5')
                plt.plot(self.t_i, new_sig[2][1])
                plt.subplot(426)
                plt.title('Wavelet coefficient 6')
                plt.plot(self.t_i, new_sig[1][1])
                plt.subplot(427)
                plt.title('Wavelet coefficient 7')
                plt.plot(self.t_i, new_sig[0][1])

            # Get peaks-locs, compute interval
            loc_idx = signal.find_peaks_cwt(new_sig[num_level - slct_lvl][1], np.arange(1, 10))
            if show:
                # Plot slected wavelet coefficient peaks
                plt.subplot(420 + slct_lvl)
                plt.plot(self.t_i[loc_idx], new_sig[num_level - slct_lvl][1][loc_idx])
                plt.pause(0.000001)
            peaks = {'Time': loc_idx * 1.0 / 20.0}
            peaks_df = pd.DataFrame(peaks)

            # Compute mean interval and get heart rate with sliding window
            hr = (60.0 / (peaks_df.diff().mean().values)).flatten()
            hr_time = peaks_df.values.flatten()

            self.hr_kinect.append(hr)
            self.kinect_time.append(hr_time)

            if show:
                # Show analysis
                plt.figure()
                plt.plot(hr_time, hr)
                plt.legend(['Kinect measurement', 'ECG Ground truth'])
                plt.pause(0.000001)


    # STFT processing and show #
    def stft_proc(self, sig, show=True):
        specgram, t, z = signal.stft(sig, 20, nperseg=200)

        if show:
            plt.figure()
            plt.pcolormesh(t, specgram, np.abs(z))
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.pause(0.000001)


if __name__ == '__main__':
    data = OnlineProcess(pickled_file='CHAIR_OTIS_SAMUEL_2018_03_22_16_06.p',
                         ref_file='REF_CHAIR_OTIS_SAMUEL_2018_03_22_16_06.mat', show=False)
    data.wvt_proc([data.interp_x], show=False)

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
