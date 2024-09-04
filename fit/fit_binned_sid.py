
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import astropy
import logging
import re

from astropy.visualization import astropy_mpl_style, quantity_support
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 360
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
        logging.basicConfig(filename='fit_icecube.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit_icecube')
        print(f'-> Reading {filename.split("_")[2][:-4]} MHz')

    def read_npz(self, filename):
        data = np.load(filename, allow_pickle=True) #* allow pickling -> use of packed objects
        time = data['time']
        t = np.array([val.timestamp() for val in time])
        self.time = self.get_sidereal_time(t)
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

    def set_window_time(self, init_time, final_time, freq):
        self.init_time, self.final_time = init_time, final_time
        init_time, final_time, self.freq = datetime.strptime(init_time, "%Y-%m-%d"), datetime.strptime(final_time, "%Y-%m-%d"), freq
        self.dataframe = self.dataframe.sort_values('time', axis=0, ascending=True)
        self.dataframe = self.dataframe.loc[self.dataframe['time'] > init_time]
        self.dataframe = self.dataframe.loc[self.dataframe['time'] < final_time]

    def get_sidereal_time(self, time_utc):
        #print(time_utc)
        location = astropy.coordinates.EarthLocation.of_site("IceCube")
        time = astropy.time.Time(time_utc, format='unix', location=location)
        sidereal_time = time.sidereal_time('apparent').value
        return sidereal_time

    def bin_data(self, time, rms, bins=100):
        #? now use sem instead of std
        std, _, _ = binned_statistic(time, rms, statistic='std', bins=bins)
        n, _, _ = binned_statistic(time, rms, statistic='count', bins=bins)
        sem = std/np.sqrt(n)
        bins, bin_edges, _ = binned_statistic(time, rms, statistic=self.stat, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        return bins, bin_centers, sem

    def chi_squared(self, y_obs, y_fit, sigma):
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-5)

    def fit_sin(self, time, rms, ant_i, chan_i):
        t = np.array([val.timestamp() for val in time]) #* converts datetime to a unix timestamp
        #t_sid = self.get_sidereal_time(t)
        #rms = savgol_filter(rms, 7000, 1)
        #! initial values to fit
        a, mean = 0, 0
        phase, phase2 = 0, 0
        amp, amp2 = np.std(rms), np.std(rms)
        sidereal_freq = 1/(23*60*60 + 56*60 + 4.092) #* sidereal freq
        solar_freq = 1/(24*60*60) #* solar freq

        def fit_function(t, amp, phase, amp2, phase2, a, mean):
            return amp * np.sin(2 * np.pi * sidereal_freq * t + phase) + amp2 * np.sin(2 * np.pi * solar_freq * t + phase2) + a*t# + mean

        initial_guess = [amp, phase, amp2, phase2, a, mean]
        param_bounds = ([0, -np.inf, 0, -np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        bins, bin_centers, sem = self.bin_data(t, rms) #* bin before fitting

        valid_bins = ~np.isnan(bins) & ~np.isnan(bin_centers) #? filter out bins with NaN or Inf values
        fit_bin_centers, fit_bins = bin_centers[valid_bins], bins[valid_bins]

        popt, pcov = curve_fit(fit_function, fit_bin_centers, fit_bins, p0=initial_guess, bounds=param_bounds)
        perr = np.sqrt(np.diag(pcov)) #* std of the fit params
        amp, phase, amp2, phase2, a, mean = popt #* fit values
        amp_std, phase_std, amp2_std, phase2_std, a_std, mean_std = perr #* std of fit values
        fit = fit_function(bin_centers, *popt)

        #self.logger.info({'freq':self.freq, 'ant':ant_i, 'pol':chan_i, 'init_time':self.init_time, 'final_time':self.final_time, 'stat':self.stat})
        #self.logger.info({'A1':amp, 'phi1':phase, 'A2':amp2, 'phi2':phase2, 'mean':mean})
        #self.logger.info({'A1_std':amp_std, 'phi1_std':phase_std, 'A2_std':amp2_std, 'phi2_std':phase2_std, 'mean_std':mean_std})

        print(f'Chi-squared: {self.chi_squared(rms, fit_function(t, *popt), np.std(fit)):.3f}')
        print(f'SEM: {np.std(fit)/len(fit):.3f}')
        print(f'A_1: {amp:.3f} ± {amp_std:.3f}')
        print(f'phi_1: {phase:.3f} ± {phase_std:.3f}')
        print(f'mean: {mean:.3f} ± {mean_std:.3f}')

        return fit, bin_centers, sem #* final fit


    def plot_fit(self, ax, time, data_fit, **kwargs):
        ax.plot_date(time, data_fit, **kwargs)

    def plot_pol(self, ax, ant_i, chan_i):
        color, color2, color3 = ['y', 'c'], ['mediumseagreen', 'cornflowerblue'], ['darkgreen', 'navy']
        id_i = f'{ant_i+1}{chan_i}'
        print(f'\n- Ant. {ant_i+1}, Pol. {chan_i+1}')

        average_rms = self.dataframe['average'+id_i] - np.mean(self.dataframe['average'+id_i]) #? center data
        time = self.dataframe['time']
        rms_cent = self.dataframe['rms'+id_i] - np.mean(self.dataframe['rms'+id_i])

        start_time, end_time = datetime.strptime(self.init_time, "%Y-%m-%d"), datetime.strptime(self.final_time, "%Y-%m-%d")
        mask = (time >= start_time) & (time <= end_time)
        time, rms_cent, average_rms = time[mask], rms_cent[mask], average_rms[mask]

        data_fit, bin_centers, sem = self.fit_sin(time, average_rms, ant_i, chan_i)

        #ax.plot_date(time, rms_cent, fmt=',', c=color[chan_i], label=f'Ant. {ant_i+1}, Pol. {chan_i}')
        ax.plot_date(time, average_rms, fmt=',', c=color2[chan_i], label=f'Moving avg Ant. {ant_i+1}, Ch. {chan_i+1}')

        bin_centers_dates = [datetime.fromtimestamp(tc) for tc in bin_centers]
        self.plot_fit(ax, bin_centers_dates, data_fit, lw=1.4, fmt='-.', label=f'Fit rms Ant. {ant_i+1}, Ch. {chan_i+1}', c=color3[chan_i])
        #ax.errorbar(bin_centers_dates, data_fit, yerr=sem, markersize=1.0, fmt ='o', label=f'SEM Ant. {ant_i+1}, Ch. {chan_i+1}', ecolor=color3[chan_i], markeredgecolor=color3[chan_i], markerfacecolor=color3[chan_i])

        t = np.array([val.timestamp() for val in time])
        x = self.bin_data(t, average_rms)[0]-data_fit
        mse = np.mean((x[~np.isnan(x)])**2)
        print(f'mse: {mse:.3f}')


    def plot_rms(self):
        fig = plt.figure(figsize=[25, 16])
        fig.tight_layout(pad=1)
        spec = gridspec.GridSpec(ncols=1, nrows=3)
        self.plot_pol(fig.add_subplot(spec[1]), 1, 1)
        '''for ant_i in range(self.ant):
            ax = fig.add_subplot(spec[ant_i])
            for chan_i in range(self.pol):
                self.plot_pol(ax, ant_i, chan_i)
            ax.set_ylabel('RMS')
            ax.set_ylim(-2,2)
            plt.legend(loc='upper left')
            plt.setp(ax.get_xticklabels(), visible=True)'''

        plt.suptitle(f'Frequency: {self.freq} MHz\n From {self.init_time} to {self.final_time}\nBinned using {self.stat}', fontsize=18, y=0.95)
        plt.savefig(f'../plots_icecube/fit_{self.stat}_{self.init_time}_{self.final_time}_{self.freq}.png')
        plt.close()
        self.logger.info('-')


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='.npz file name', required=True)
parser.add_argument('--stat', '-st', type=str, help='mean or median', required=False, default='mean')
parser.add_argument('--init_time', '-ini', type=str, help='initial time -> YYYY-MM-DD', required=False)
parser.add_argument('--final_time', '-fin', type=str, help='final time -> YYYY-MM-DD', required=False)
args = parser.parse_args()

if not args.init_time or not args.final_time:
    match = re.search(r'(\d{4}_\d{2}_\d{2})-(\d{4}_\d{2}_\d{2})', args.file)
    init_time = match.group(1).replace('_', '-')
    final_time = match.group(2).replace('_', '-')
else:
    init_time = args.init_time
    final_time = args.final_time

fit = fit_data(f'/data/user/valeriatorres/galactic_noise/SouthPole/{args.file}', args.stat)
fit.process_data()
fit.set_window_time(init_time, final_time, args.file.split('_')[-1][:-4])
fit.plot_rms()