####
# FOR USE WITH PYTHON 3
###


import pickle
import matplotlib.pyplot as plt
from scipy import signal, interpolate, io
import numpy as np
import pywt
import math
import pandas as pd
import biosppy as bp
import peakutils as pk
import os


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


class OfflineProcess(object):
    def __init__(self, pickled_file='CHAIR_OTIS_SAMUEL_2018_03_22_16_06.p',
                 ref_file='REF_CHAIR_OTIS_SAMUEL_2018_03_22_16_06.mat', show=True):
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

    # Data import and show #
    def data_import(self, pickle_name, refname, show=True, is_ref_mat=True):
        # Import data
        self.t_i, self.interp_x, self.interp_y, self.interp_z, self.freq, self.fft_x, self.fft_y, self.fft_z = pickle.load(
            open(pickle_name, 'rb'))
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
        if '.mat' in refname:
            mat_ecg = io.loadmat('../recordings/' + refname)
            self.ref = bp.ecg.ecg(signal=mat_ecg['data'][0], sampling_rate=mat_ecg['samplerate'][0][0], show=False)
            self.ref_time = self.ref[5]
            self.ref_hr = self.ref[6]

        else:
            self.ref = pd.DataFrame.from_csv('../recordings/' + refname)
            filter_time = self.ref.index.values <= self.largest_base / 20.0
            self.ref_hr = self.ref.values[filter_time].flatten()
            self.ref_time = self.ref.index.values[filter_time]
        if show:
            self.show_data()

    # Show data and related FFT
    # INSERT FOR LOOP FOR PLOTS
    def show_data(self):
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
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)
        plt.pause(0.000001)

    # Wavelet processing and show #
    def wvt_proc(self, show=True):
        num_level = int(np.log2(self.largest_base))
        slct_lvl = 3
        euc_dis = get_euclidean_distance(self.interp_x, self.interp_y, self.interp_z)
        for axis in [euc_dis, euc_dis, euc_dis]:
            wlt = pywt.Wavelet('db6')
            new_sig = pywt.swt(axis, wavelet=wlt, level=num_level)

            # INSERT FOR LOOP FOR PLOTS
            if show:
                plt.figure()
                for i in range(1, num_level + 1):
                    plt.subplot(4, 3, i)
                    plt.title('Wavelet coefficient ' + str(i))
                    plt.plot(self.t_i, new_sig[-i][0])
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                    wspace=0.5, hspace=0.5)
                plt.pause(0.00001)

            # Get peaks-locs, compute interval
            loc_idx = pk.indexes(new_sig[-slct_lvl][0], min_dist=6, thres=0)
            if show:
                # Plot slected wavelet coefficient peaks
                plt.subplot(4, 3, slct_lvl)
                plt.plot(self.t_i[loc_idx], new_sig[-slct_lvl][0][loc_idx], 'r*')
                plt.pause(0.000001)
            peaks = {'Time': loc_idx * 1.0 / 20.0}
            peaks_df = pd.DataFrame(peaks)

            # Compute mean interval and get heart rate with sliding window
            hr = (60.0 / (peaks_df.diff().rolling(10).mean().values)).flatten()
            hr_time = peaks_df.values.flatten()

            # Fit data and ref together
            avoid_nan = ~np.isnan(hr)
            hr = hr[avoid_nan]
            hr_time = hr_time[avoid_nan]
            self.ref_hr = self.ref_hr[self.ref_time >= hr_time[0]]
            self.ref_time = self.ref_time[self.ref_time >= hr_time[0]]
            hr = hr[hr_time >= self.ref_time[0]]
            hr_time = hr_time[hr_time >= self.ref_time[0]]

            # Get error with ref
            interp_ref_f = interpolate.interp1d(self.ref_time, self.ref_hr)
            interp_ref = interp_ref_f(hr_time)
            # Basic statistical analysis
            error = abs(interp_ref - hr)
            error_m = np.mean(error)
            error_std = np.std(error)
            m, b = np.polyfit(interp_ref, hr, 1)

            self.hr_kinect.append(hr)
            self.kinect_time.append(hr_time)

            if show:
                # Show analysis
                plt.figure()
                plt.plot(hr_time, hr)
                plt.plot(hr_time, interp_ref)
                plt.legend(['Kinect measurement', 'ECG Ground truth'])
                plt.pause(0.000001)
                plt.figure()
                plt.plot(hr_time, error)
                plt.pause(0.000001)
                # plt.figure()
                # plt.plot(interp_ref, hr,'*')
                # plt.plot(interp_ref, m * interp_ref + b, '-')
                # plt.pause(0.000001)

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
    data = OfflineProcess()
    filepath = '../recordings/2nd take/centroids/'
    refpath = '../recordings/2nd take/refs/'
    files = sorted([f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))])
    refs = sorted([f for f in os.listdir(refpath) if os.path.isfile(os.path.join(refpath, f))])
    for ref, file in zip(refs, files):
        data.data_import(pickle_name=filepath + file,
                         refname=refpath + ref, show=False)
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
        hr = np.array([[hr_z]])  # np.array([hr_x, hr_y, hr_z])
        mean_hr = np.mean(hr, axis=0)

        # Plot results
        plt.figure()
        plt.plot(data.ref_time, data.ref_hr)
        plt.plot(data.ref_time, hr_x)
        plt.xlabel('Temps (s)')
        plt.ylabel('Rythme cardiaque (BPM)')
        plt.legend(['Référence', 'Kinect'])  # ['Ground  truth', 'HR z axis']
        plt.pause(0.000001)
        # plt.figure()
        # plt.plot(data.ref_time, data.ref_hr)
        # plt.plot(data.ref_time, mean_hr)
        # plt.pause(0.000001)
        # plt.figure()
        # plt.plot(data.ref_time, abs(data.ref_hr - mean_hr))
        # plt.pause(0.000001)
        abs_error = abs(data.ref_hr - mean_hr)
        mean = np.mean(abs_error)
        dev = np.std(abs_error)
        name = file[:-2] + '_z_AE'
        # pickle.dump(abs_error,
        #             open('../recordings/2nd take/centroids/results/db6/euclidean/'+name + '.p', 'wb'))

    assert True
