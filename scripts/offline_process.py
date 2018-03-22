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


class OfflineProcess(object):
    def __init__(self, pickled_file, ref_file, show=True):
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
        # Select ref type to open
        is_ref_mat = True if '.mat' in ref_file else False
        self.data_import(pickled_file, ref_file, show, is_ref_mat)

    # Data import and show #
    def data_import(self, pickle_name, refname, show=True, is_ref_mat=True):
        # Import data
        self.t_i, self.interp_x, self.interp_y, self.interp_z, self.freq, self.fft_x, self.fft_y, self.fft_z = pickle.load(
            open('../recordings/'+pickle_name + '.p', 'rb'))
        # Cut data to magnitude of 2
        self.largest_base = floor_log(len(self.t_i), 2)

        self.t_i = self.t_i[:self.largest_base]
        self.interp_x = self.interp_x[:self.largest_base]
        self.interp_y = self.interp_y[:self.largest_base]
        self.interp_z = self.interp_z[:self.largest_base]
        self.freq = self.freq[:self.largest_base]
        self.fft_x = self.fft_x[:self.largest_base]
        self.fft_y = self.fft_y[:self.largest_base]
        self.fft_z = self.fft_z[:self.largest_base]
        if is_ref_mat:
            mat_ecg = io.loadmat('../recordings/'+refname)
            self.ref = bp.ecg()
        else:
            self.ref = pd.DataFrame.from_csv('../recordings/'+refname)
            filter_time = self.ref.index.values <= self.largest_base / 20.0
            self.ref_hr = self.ref.values[filter_time].flatten()
            self.ref_time = self.ref.index.values[filter_time]

        if show:
            # Plot real and interpolated signal
            plt.figure()
            plt.subplot(321)
            plt.title('X motion magnitude')
            plt.xlabel('Time (s)')
            plt.ylabel('Motion (m)')
            plt.plot(self.t_i, self.interp_x, '-r')
            plt.subplot(322)
            plt.title('X Fourier transform')
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude x')
            plt.plot(self.freq, self.fft_x)

            plt.subplot(323)
            plt.title('Y motion magnitude')
            plt.xlabel('Time (s)')
            plt.ylabel('Motion (m)')
            plt.plot(self.t_i, self.interp_y, '-r')
            plt.subplot(324)
            plt.title('Y Fourier transform')
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude y')
            plt.plot(self.freq, self.fft_y)

            plt.subplot(325)
            plt.title('Z motion magnitude')
            plt.xlabel('Time (s)')
            plt.ylabel('Motion (m)')
            plt.plot(self.t_i, self.interp_z, '-r')
            plt.subplot(326)
            plt.title('Z Fourier transform')
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude z')
            plt.plot(self.freq, self.fft_z)

            plt.pause(0.000001)

    # Wavelet processing and show #
    def wvt_proc(self, interp, show=True):
        num_level = 8
        slct_lvl = 4
        wlt = pywt.Wavelet('db6')
        new_sig = pywt.swt(interp, wavelet=wlt, level=num_level)

        if show:
            plt.figure()
            plt.subplot(421)
            plt.title('Wavelet coefficient 1')
            plt.plot(self.t_i, new_sig[7][1])
            plt.subplot(422)
            plt.title('Wavelet coefficient 2')
            plt.plot(self.t_i, new_sig[6][1])
            plt.subplot(423)
            plt.title('Wavelet coefficient 3')
            plt.plot(self.t_i, new_sig[5][1])
            plt.subplot(424)
            plt.title('Wavelet coefficient 4')
            plt.plot(self.t_i, new_sig[4][1])
            plt.subplot(425)
            plt.title('Wavelet coefficient 5')
            plt.plot(self.t_i, new_sig[3][1])
            plt.subplot(426)
            plt.title('Wavelet coefficient 6')
            plt.plot(self.t_i, new_sig[2][1])
            plt.subplot(427)
            plt.title('Wavelet coefficient 7')
            plt.plot(self.t_i, new_sig[1][1])
            plt.subplot(428)
            plt.title('Wavelet coefficient 8')
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
        hr = (60.0 / (peaks_df.diff().rolling(10).mean().values)).flatten()
        hr_time = peaks_df.values.flatten()
        avoid_nan = ~np.isnan(hr)
        # Fit data and ref together
        self.ref_hr = self.ref_hr[self.ref_time >= hr_time[avoid_nan][0]]
        self.ref_time = self.ref_time[self.ref_time >= hr_time[avoid_nan][0]]


        # Get error with ref
        interp_hr_f = interpolate.interp1d(hr_time[avoid_nan], hr[avoid_nan])
        interp_hr = interp_hr_f(self.ref_time)
        # Basic statistical analysis
        error = abs(interp_hr - self.ref_hr)
        error_m = np.mean(error)
        error_std = np.std(error)
        m, b = np.polyfit(self.ref_hr, interp_hr, 1)

        # Show analysis
        plt.figure()
        plt.plot(self.ref_time, interp_hr)
        plt.plot(self.ref_time, self.ref_hr)
        plt.pause(0.000001)
        plt.figure()
        plt.plot(self.ref_hr, interp_hr,'*')
        plt.plot(self.ref_hr, m * self.ref_hr + b, '-')
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
    data = OfflineProcess('CHAIR_OTIS_SAMUEL_2018_03_22_16_06.p', 'REF_CHAIR_OTIS_SAMUEL_2018_03_22_16_06.mat', show=False)
    data.wvt_proc(data.interp_x, show=False)
    assert True
