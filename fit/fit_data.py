from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import logging
import re

from astropy.visualization import astropy_mpl_style, quantity_support
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 800
plt.style.use(astropy_mpl_style)
quantity_support()


class fit_data:

    def __init__(self, filename):
        self.time, self.rms = [], []
        self.ant, self.pol = 3, 2 #* 3 antennas and 2 polarisations
        self.window = 150
        self.dataframe = pd.DataFrame() #* create dataframe to make easier column operations
        self.read_npz(filename) #* read compressed numpy files
        logging.basicConfig(filename='fit_ic.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit_ic')
        print(f'-> Reading {filename.split("_")[2][:-4]} MHz')

    def read_npz(self, filename):
        data = np.load(filename, allow_pickle=True) #* allow pickling -> use of packed objects
        self.time = data['time']
        self.rms = np.array([data['rms10'], data['rms11'], data['rms20'], data['rms21'], data['rms30'], data['rms31']])

    def process_data(self):
        self.to_dataframe()
        self.clean_data()
        #self.remove_outliers()
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
                
    def remove_outliers(self, z_thresh=3.0):
        #remove outliers using Z-score
        for ant_i in range(self.ant):
            for chan_i in range(self.pol):
                id_i = f'rms{ant_i+1}{chan_i}'
                rms_mean = np.mean(self.dataframe[id_i])
                rms_std = np.std(self.dataframe[id_i])
                z_scores = (self.dataframe[id_i] - rms_mean) / rms_std
                self.dataframe = self.dataframe[np.abs(z_scores) < z_thresh]

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

    def chi_squared(self, y_obs, y_fit, sigma):
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-5)

    def interpolate_data(self, time, rms):
        t = np.array([val.timestamp() for val in time])
        interp_func = interp1d(t, rms)
        t_interp = np.linspace(t.min(), t.max(), num=len(t))
        rms_interp = interp_func(t)
        return t, rms_interp

    def fit_sin(self, time, rms, ant_i, chan_i):
        t = np.array([val.timestamp() for val in time]) #* converts datetime to a unix timestamp

        #! initial values to fit
        a, mean, = 0.0, np.mean(rms)
        amp, phase, freq = 0.0, 0.0, 0.0

        def fit_function(t, amp, freq, phase, a, mean):
            return amp * np.sin(2 * np.pi * freq * t + phase) + a*t + mean

        initial_guess = [amp, freq,  phase, a, mean]
        param_bounds=([1, -np.inf, -np.inf, -np.inf, -np.inf], [2, np.inf, np.inf, np.inf, np.inf,])
        mask = ~np.isnan(rms) #? filter out bins with NaN or Inf values
        t, rms = t[mask], rms[mask]
        popt, pcov = curve_fit(fit_function, t, rms, p0=initial_guess, bounds=param_bounds)

        perr = np.sqrt(np.diag(pcov)) #* std of the fit params
        amp, freq,  phase, a, mean = popt #* fit values
        amp_std, freq_std,  phase_std, a_std, mean_std = perr #* std of fit values
        data_fit = fit_function(t, *popt) #* final fit

        #self.logger.info({'freq':self.freq, 'ant':ant_i, 'pol':chan_i, 'init_time':self.init_time, 'final_time':self.final_time})
        #self.logger.info({'A1':amp, 'phi1':phase, 'A2':amp2, 'phi2':phase2, 'mean':mean})
        #self.logger.info({'A1_std':amp_std, 'phi1_std':phase_std, 'A2_std':amp2_std, 'phi2_std':phase2_std, 'mean_std':mean_std})

        #print(f'Chi-squared: {self.chi_squared(rms, fit_function(t, *popt), np.std(fit)):.3f}')
        #print(f'SEM: {np.std(fit)/len(fit):.3f}')
        print(f'A: {amp:.3f} ± {amp_std:.3f}')
        print(f'phi: {phase:.3f} ± {phase_std:.3f}')
        print(f'f: {freq:.3f} ± {freq_std:.3f}')
        print(f'a: {a:.3f} ± {a_std:.3f}')
        print(f'mean: {mean:.3f} ± {mean_std:.3f}')
        return data_fit


    def plot_fit(self, ax, time, data_fit, **kwargs):
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
        data_fit = self.fit_sin(time, average_rms, ant_i, chan_i)

        ax.plot_date(time, rms_cent, fmt=',', c=color[chan_i], alpha=0.7, label=f'Ant. {ant_i+1}, Pol. {chan_i}')
        ax.plot_date(time, average_rms, fmt=',', c=color2[chan_i], label=f'Moving avg Ant. {ant_i+1}, Ch. {chan_i+1}')
        #ax.plot_date(time, rms_interp, fmt=',', c='black')

        self.plot_fit(ax, time, data_fit, lw=1.4, fmt='--', label=f'Fit rms Ant. {ant_i+1}, Ch. {chan_i+1}', c=color3[chan_i])
        #sem = np.std(data_fit)/len(data_fit)
        #ax.errorbar(time, data_fit, yerr=sem, fmt ='o', markersize=1, label=f'SEM Ant. {ant_i+1}, Ch. {chan_i+1}', ecolor=color3[chan_i], markeredgecolor=color3[chan_i], markerfacecolor=color3[chan_i])

        #mse = np.mean((average_rms - data_fit) ** 2)
        #print(f'mse: {round(mse,3)}')


    def plot_rms(self):
        fig = plt.figure(figsize=[25, 16])
        fig.tight_layout(pad=1)
        spec = gridspec.GridSpec(ncols=1, nrows=3)

        for ant_i in range(self.ant):
            ax = fig.add_subplot(spec[ant_i])
            for chan_i in range(self.pol):
                self.plot_pol(ax, ant_i, chan_i)
            #self.plot_pol(ax, ant_i, 0)
            ax.set_ylabel('RMS')
            ax.set_ylim(-2,2)
            plt.legend(loc='upper left')
            plt.setp(ax.get_xticklabels(), visible=True)

        plt.suptitle(f'Frequency: {self.freq} MHz\n From {self.init_time} to {self.final_time}', fontsize=18, y=0.95)
        #plt.savefig(f'fit_{self.init_time}_{self.final_time}_{self.freq}.png')
        plt.savefig('fit_norm.png')
        plt.close()
        self.logger.info('-')


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='.npz file name', required=True)
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

fit = fit_data(f'/data/user/valeriatorres/galactic_noise/SouthPole/{args.file}')
fit.process_data()
fit.set_window_time(init_time, final_time, args.file.split('_')[-1][:-4])
fit.plot_rms()