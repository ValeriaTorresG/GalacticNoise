from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from astropy.visualization import astropy_mpl_style, quantity_support
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 2000
plt.style.use(astropy_mpl_style)
quantity_support()


class fit_data:

    def __init__(self, filename, stat):
        self.time, self.rms = [], []
        self.ant, self.pol = 3, 2 #* 3 antennas and 2 polarisations
        self.window = 150
        self.stat = stat
        self.dataframe = pd.DataFrame() #* create dataframe to make easier column operations
        self.read_npz(filename) #* read compressed numpy files
        logging.basicConfig( filename='fit.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit')
        print(f'-> Reading {filename.split("_")[2][:-4]} MHz')

    def read_npz(self, filename):
        data = np.load(filename, allow_pickle=True) #* allow pickling -> use of packed objects
        self.time = data['time']
        self.rms = np.array([data['rms10'], data['rms11'], data['rms20'], data['rms21'], data['rms30'], data['rms31']])

    def process_data(self):
        self.to_dataframe()
        self.clean_data()
        self.get_moving_avg()
        print('\n------ Fitting data ------')

    def to_dataframe(self):
        self.dataframe['time'] = self.time
        for ant_i in range(self.ant):
            for chan_i in range(self.pol):
                id_i = f'rms{ant_i+1}{chan_i}'
                self.dataframe[id_i] = self.rms[2*ant_i + chan_i]

    def clean_data(self, n=17): #? n to calc a threshold by adding n to the minimum rms.
        for ant_i in range(self.ant):
            for chan_i in range(self.pol):
                id_i = f'rms{ant_i+1}{chan_i}'
                self.dataframe = self.dataframe[self.dataframe[id_i] <= np.min(self.dataframe[id_i]) + n] #* remove outliers

    def get_moving_avg(self): #* calc moving avg using rolling
        for ant_i in range(self.ant):
            for chan_i in range(self.pol):
                id_i = f'{ant_i+1}{chan_i}'
                self.dataframe['average'+id_i] = self.dataframe['rms'+id_i].rolling(self.window).mean()

    def set_window_time(self, init_time, final_time):
        self.init_time, self.final_time = init_time, final_time
        self.dataframe = self.dataframe.sort_values('time', axis=0, ascending=True)
        self.dataframe = self.dataframe.loc[self.dataframe['time'] > init_time]
        self.dataframe = self.dataframe.loc[self.dataframe['time'] < final_time]

    def bin_data(self, time, rms, bins=100):
        #? now use sem instead of std
        std, _, _ = binned_statistic(time, rms, statistic='std', bins=bins)
        n, _, _ = binned_statistic(time, rms, statistic='count', bins=bins)
        sem = std/np.sqrt(n)

        bins, bin_edges, _ = binned_statistic(time, rms, statistic=self.stat, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

        valid_bins = ~np.isnan(bins) & ~np.isnan(bin_centers) #? filter out bins with NaN or Inf values
        bin_centers = bin_centers[valid_bins]
        bins = bins[valid_bins]

        return bins, bin_centers, sem

    def chi_squared(self, y_obs, y_fit, sigma):
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-5)

    def fit_sin(self, time, rms):
        t = np.array([val.timestamp() for val in time]) #* converts datetime to a unix timestamp

        #! initial values to fit
        mean, std, std2 = 0, np.std(rms), np.std(rms)
        phase, phase2 = 0, 0.5
        amp, amp2 = std/np.sqrt(2), 0.001
        sidereal_freq = 1/(23*60*60 + 56*60 + 4.092) #* sidereal freq
        solar_freq = 1/(24*60*60) #* solar freq

        def fit_function(t, amp, phase, amp2, phase2, mean):
            return amp * np.sin(2 * np.pi * sidereal_freq * t + phase) + amp2 * np.sin(2 * np.pi * solar_freq * t + phase2) + mean

        initial_guess = [amp, phase, amp2, phase2, mean]
        param_bounds=([0, -np.inf, 0, -np.inf, -np.inf],[np.inf, np.inf, np.inf, np.inf, np.inf])
        bins, bin_centers, _ = self.bin_data(t, rms) #* bin before fitting
        popt, pcov = curve_fit(fit_function, bin_centers, bins, p0=initial_guess, bounds=param_bounds)

        perr = np.sqrt(np.diag(pcov)) #* std of the fit params
        amp, phase, amp2, phase2, mean = popt #* fit values
        amp_std, phase_std, amp2_std, phase2_std, mean_std = perr #* std of fit values

        print(f'Chi-squared: {self.chi_squared(rms, fit_function(t, *popt), std):.3f}')
        print(f'SEM: {std/len(rms):.3f}')

        print('Parameters:')
        print(f'A_1: {amp:.3f} ± {amp_std:.3f}')
        print(f'phi_1: {phase:.3f} ± {phase_std:.3f}')
        print(f'mean: {mean:.3f} ± {mean_std:.3f}')
        print(f'A_2: {amp2:.3f} ± {amp2_std:.3f}')
        print(f'phi_2: {phase2:.3f} ± {phase2_std:.3f}')
        return fit_function(t, *popt), bins #* final fit


    def plot_fit(self, ax, time, data_fit, bins, **kwargs):
        ax.plot_date(time, data_fit, **kwargs)

    def plot_pol(self, ax, ant_i, chan_i):
        color, color2, color3 = ['y', 'c'], ['mediumseagreen', 'cornflowerblue'], ['darkgreen', 'navy']
        id_i = f'{ant_i+1}{chan_i}'
        print(f'\n- Ant. {ant_i+1}, Pol. {chan_i+1}')

        average_rms = self.dataframe['average'+id_i] - np.mean(self.dataframe['average'+id_i]) #? center data
        time = self.dataframe['time']
        rms_cent = self.dataframe['rms'+id_i] - np.mean(self.dataframe['rms'+id_i])

        start_time, end_time = self.init_time, self.final_time
        mask = (time >= start_time) & (time <= end_time)
        time, rms_cent, average_rms = time[mask], rms_cent[mask], average_rms[mask]
        data_fit, bins = self.fit_sin(time, average_rms)

        ax.plot_date(time, rms_cent, fmt=',', c=color[chan_i], alpha=0.1, label=f'Ant. {ant_i+1}, Pol. {chan_i}')
        ax.plot_date(time, average_rms, fmt=':', alpha=0.7, c=color2[chan_i], label=f'Moving avg Ant. {ant_i+1}, Ch. {chan_i+1}')

        self.plot_fit(ax, time, data_fit, bins, lw=2, fmt='--', label=f'Fit rms Ant. {ant_i+1}, Ch. {chan_i+1}', c=color3[chan_i])
        mse = np.mean((average_rms-data_fit)**2)
        print(f'mse: {mse:.3f}')


    def plot_rms(self):
        fig = plt.figure(figsize=[25, 16])
        #fig.tight_layout(pad=1)
        spec = gridspec.GridSpec(ncols=1, nrows=3)

        for ant_i in range(self.ant):
            ax = fig.add_subplot(spec[ant_i])
            for chan_i in range(self.pol):
                self.plot_pol(ax, ant_i, chan_i)
            ax.set_ylabel('RMS')
            ax.set_ylim(-4,8)
            plt.legend(loc='upper left')
            plt.setp(ax.get_xticklabels(), visible=True)
            plt.suptitle(f'Using {self.stat}', fontsize=18, y=0.95)

        plt.savefig(f'fit_{self.stat}.png')
        plt.close()


filename = 'GalOscillation_Deconvolved_140-190.npz'
startTime = '2023-05-05'
endTime = '2023-05-15'
for stat in ['mean', 'median']:
    fit = fit_data(filename, stat)
    fit.process_data()
    startTime = '2023-05-05'
    endTime = '2023-05-15'
    fit.set_window_time(startTime, endTime)
    fit.plot_rms()