from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from astropy import units as u
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import logging
import astropy
import time
import re

from astropy.visualization import astropy_mpl_style, quantity_support
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 360
plt.style.use(astropy_mpl_style)
quantity_support()

#python fit_binned_sid.py -f GalOscillation_Time_2024_01_05-2024_01_26_Freq_70-150.npz -ini 2024-01-05 -fin 2024-01-09

class fit_data:

    def __init__(self, filename, stat, time):
        self.time, self.rms = [], []
        self.ant, self.pol = 3, 2 #* 3 antennas and 2 polarisations
        self.window = 150
        self.stat, self.use_time = stat, time
        self.dataframe = pd.DataFrame() #* create dataframe to make easier column operations
        self.read_npz(filename) #* read compressed numpy files
        logging.basicConfig(filename='fit_icecube.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit_icecube')
        print(f'-> Reading {filename.split("_")[2][:-4]} MHz')


    def read_npz(self, filename):
        data = np.load(filename, allow_pickle=True) #* allow pickling -> use of packed objects
        self.time = data['time']
        self.rms = np.array([data['rms10'], data['rms11'], data['rms20'], data['rms21'], data['rms30'], data['rms31']])


    def clean_outliers(self, threshold=5.0): #! remove outliers that are more than n standard deviations from the mean
        for ant_i in range(self.ant):
            for chan_i in range(self.pol):
                id_i = f'rms{ant_i+1}{chan_i}'
                rms = self.dataframe[id_i]
                mean = np.mean(rms)
                std = np.std(rms)
                self.dataframe = self.dataframe[(rms - mean) <= threshold * std]


    def process_data(self):
        self.to_dataframe()
        self.clean_data()
        self.get_moving_avg()
        self.clean_outliers()
        print(f'\n{"-"*6} Fitting data {"-"*6}')


    def to_dataframe(self):
        self.dataframe['time'] = self.time
        for ant_i in range(self.ant):
            for chan_i in range(self.pol):
                id_i = f'rms{ant_i+1}{chan_i}'
                self.dataframe[id_i] = self.rms[2*ant_i + chan_i]


    def clean_data(self, n=50): #? n to calc a threshold by adding n to the minimum rms.
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
        location = astropy.coordinates.EarthLocation.of_site("IceCube")
        time = astropy.time.Time(time_utc, location=location)
        sidereal_time = time.sidereal_time('apparent').value
        return sidereal_time


    def bin_data(self, time, rms):
        delta = datetime.strptime(self.final_time, "%Y-%m-%d") - datetime.strptime(self.init_time, "%Y-%m-%d")
        bins = delta.days*100
        bins, bin_edges, _ = binned_statistic(time, rms, statistic=self.stat, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        mask = np.abs(bins)<3 & ~np.isnan(bins)
        bin_centers, bins = bin_centers[mask], bins[mask]
        return bins, bin_centers


    def chi_squared(self, y_obs, y_fit, sigma):
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-3)


    def fit_sin(self, time, rms, ant_i, chan_i):
        t = self.get_sidereal_time(time)

        bins, bin_centers = self.bin_data(t, rms)

        guess_mean, guess_std = np.mean(bins),  np.std(bins)
        guess_phase, guess_freq = 1.0, 0.00002314814 * 6
        guess_amp = guess_std/np.sqrt(2)

        bin_centers = np.asarray(bin_centers)
        data_first_guess = guess_std * np.sin(t * guess_freq + guess_phase) + guess_mean

        def optimize_func(x): return x[0] * np.sin(x[1] * bin_centers + x[2]) + x[3] - bins
        est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean], maxfev=100000)[0]

        data_fit = est_amp * np.sin(est_freq * bin_centers + est_phase) + est_mean
        param = f'A: {est_amp:.3f}\nf: {est_freq*60*60*24:.3f}, {1/est_freq:.3f}\nphi: {est_phase:.3f}, {np.rad2deg(est_phase%(np.pi)):.3f} deg\nmean: {est_mean:.3f}'
        self.logger.info({'freq':self.freq, 'time':self.use_time, 'ant':ant_i, 'pol':chan_i, 'init_time':self.init_time, 'final_time':self.final_time, 'stat':self.stat})
        self.logger.info({'amp':est_amp, 'freq':est_freq, 'phase':est_phase, 'mean':est_mean})

        return bin_centers, data_fit, param


    def plot_pol(self, ax, ant_i, chan_i):
        color, color2, color3 = ['y', 'c'], ['mediumseagreen', 'cornflowerblue'], ['darkgreen', 'navy']
        id_i = f'{ant_i+1}{chan_i}'
        print(f'\n- Ant. {ant_i+1}, Pol. {chan_i+1}')

        time = self.dataframe['time']
        average_rms = self.dataframe['average'+id_i] - np.mean(self.dataframe['average'+id_i]) #? center data
        rms_cent = self.dataframe['rms'+id_i] - np.mean(self.dataframe['rms'+id_i])

        sidereal_times = np.array([self.get_sidereal_time(t) for t in time])
        unique_days = pd.to_datetime(self.dataframe['time']).dt.date.unique()
        cmap = sns.color_palette("mako_r", as_cmap=True)
        colors = cmap(np.linspace(0.2, 0.8, len(unique_days)))
        l_styles = ['-', '--', '-.', ':']
        for i, day in enumerate(unique_days):
            day_mask = pd.to_datetime(self.dataframe['time']).dt.date == day
            day_sidereal_times = sidereal_times[day_mask]
            time_i = time[day_mask]
            day_rms_cent = rms_cent[day_mask]
            day_average_rms = average_rms[day_mask]
            ax.scatter(day_sidereal_times, day_rms_cent, alpha=0.7, s=0.6, color=colors[i])
            ax.scatter(day_sidereal_times, day_average_rms, color=colors[i], s=1.5, alpha=0.4)
            x_interpolated, y_interpolated, _ = self.fit_sin(time_i, day_average_rms, ant_i, chan_i)
            ax.plot(x_interpolated, y_interpolated, lw=3.5, ls=l_styles[i], label=f'{day}', color=colors[i])

        x_interpolated, y_interpolated, param = self.fit_sin(time, average_rms, ant_i, chan_i)
        sem = np.std(y_interpolated)/np.sqrt(len(y_interpolated))
        ax.plot(x_interpolated, y_interpolated, lw=4.5, label=f'Fit rms Ant. {ant_i+1}, Ch. {chan_i+1}', c='black')
        ax.errorbar(x_interpolated, y_interpolated, yerr=sem, markersize=3.0, fmt ='o', ecolor='black', markeredgecolor='black', markerfacecolor='black')

        x = self.get_sidereal_time(time)
        y = self.bin_data(x, average_rms)
        print(param)
        print(f'mse: {np.mean((y-y_interpolated)**2)}')
        print(f'Chi-squared: {self.chi_squared(y, y_interpolated, np.std(y_interpolated)):.3f}')


    def plot_rms(self):
        for chan_i in range(self.pol):
            print(f'\nPolarisation {chan_i+1}')
            fig = plt.figure(figsize=[25, 16])
            fig.tight_layout(pad=1)
            spec = gridspec.GridSpec(ncols=1, nrows=3)
            for ant_i in range(self.ant):
                ax = fig.add_subplot(spec[ant_i])
                self.plot_pol(ax, ant_i, chan_i)
                ax.set_xlabel('Sidereal time')
                ax.set_ylabel('RMS')
                ax.set_ylim(-2,2)
                plt.legend(loc='upper left')
                plt.setp(ax.get_xticklabels(), visible=True)
            plt.suptitle(f'Polarisation {chan_i+1}, frequency band: {self.freq} MHz\n From {self.init_time} to {self.final_time}\nBinned using {self.stat}', fontsize=18, y=0.95)
            plt.savefig(f'../plots/plots_icecube/fit_pol{chan_i+1}_{self.stat}_{self.use_time}_{self.init_time}_{self.final_time}_{self.freq}.png', dpi=360)
            plt.close()
        self.logger.info('-')


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='.npz file name', required=True)
parser.add_argument('--stat', '-st', type=str, help='mean or median', required=False, default='mean')
parser.add_argument('--init_time', '-ini', type=str, help='initial time -> YYYY-MM-DD', required=False)
parser.add_argument('--final_time', '-fin', type=str, help='final time -> YYYY-MM-DD', required=False)
parser.add_argument('--use_time', '-t', type=str, help='sid or utc', required=False, default='sid')
args = parser.parse_args()

if not args.init_time or not args.final_time:
    match = re.search(r'(\d{4}_\d{2}_\d{2})-(\d{4}_\d{2}_\d{2})', args.file)
    init_time = match.group(1).replace('_', '-')
    final_time = match.group(2).replace('_', '-')
else:
    init_time = args.init_time
    final_time = args.final_time

init = time.time()
fit = fit_data(f'/data/user/valeriatorres/galactic_noise/SouthPole/{args.file}', args.stat, args.use_time)
fit.process_data()
fit.set_window_time(init_time, final_time, args.file.split('_')[-1][:-4])
fit.plot_rms()
print(f'{"-"*10}> time elapsed: {(time.time()-init)/60:.3f} min')