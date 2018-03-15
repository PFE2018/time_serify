import pickle
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pywt
import math
import pandas as pd


def floor_log(num, base):
    if num < 0:
        raise ValueError("Non-negative number only.")

    if num == 0:
        return 0

    return base ** int(math.log(num, base))


class OfflineProcess(object):
    def __init__(self, pickled_file, show=True):
        self.t_i = []
        self.interp_x = []
        self.interp_y = []
        self.interp_z = []
        self.freq = []
        self.fft_x = []
        self.fft_y = []
        self.fft_z = []
        self.data_import(pickled_file, show)

    # Data import and show #
    def data_import(self, pickle_name, show=True):
        # Import data
        self.t_i, self.interp_x, self.interp_y, self.interp_z, self.freq, self.fft_x, self.fft_y, self.fft_z = pickle.load(
            open(pickle_name + '.p', 'rb'))
        # Cut data to magnitude of 2
        largest_base = floor_log(len(self.t_i), 2)

        self.t_i = self.t_i[:largest_base]
        self.interp_x = self.interp_x[:largest_base]
        self.interp_y = self.interp_y[:largest_base]
        self.interp_z = self.interp_z[:largest_base]
        self.freq = self.freq[:largest_base]
        self.fft_x = self.fft_x[:largest_base]
        self.fft_y = self.fft_y[:largest_base]
        self.fft_z = self.fft_z[:largest_base]

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
        wlt = pywt.Wavelet('db6')
        new_sig = pywt.swt(interp, wavelet=wlt, level=num_level)

        # Get peaks-locs, compute interval

        slct_lvl = 4
        loc_idx = signal.find_peaks_cwt(new_sig[num_level-slct_lvl][1], np.arange(1, 10))
        peaks = {'Time': loc_idx * 1.0 / 20.0}
        peaks_df = pd.DataFrame(peaks)
        hr = 60.0/(peaks_df.diff().rolling(20).mean().values)
        hr_time = peaks_df.values
        # Compute mean interval
        plt.figure()
        plt.plot(hr_time, hr)
        plt.pause(0.000001)
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

            # Plot slected wavelet coefficient peaks
            plt.subplot(420+slct_lvl)
            plt.plot(self.t_i[loc_idx], new_sig[num_level-slct_lvl][1][loc_idx])
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
    data = OfflineProcess('sam_still_300s_12-03-201_take2', show=False)
    data.wvt_proc(data.interp_x)
