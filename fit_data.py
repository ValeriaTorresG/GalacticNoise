from scipy.optimize import curve_fit
from datetime import datetime
import pandas as pd
import numpy as np

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

    def get_chi_squared(self, data_fit, rms):
        chi_squared = np.sum(((data_fit - rms) / data_fit) ** 2)
        red_chi_squared = chi_squared/(len(rms)-4)
        return red_chi_squared

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
        popt, pcov = curve_fit(fit_function, t, rms, p0=initial_guess, bounds=param_bounds)

        perr = np.sqrt(np.diag(pcov)) #* std of the fit params
        amp, phase, amp2, phase2, mean = popt #* fit values
        amp_std, phase_std, amp2_std, phase2_std, mean_std = perr #* std of fit values

        data_fit = fit_function(t, amp, phase, amp2, phase2, mean) #* final fit
        red_chi_squared = self.get_chi_squared(data_fit, rms)
        print(f'Reduced chi-squared: {red_chi_squared}')
        print('Parameters:')
        print(f'A_1: {round(amp, 3)} ± {round(amp_std, 3)}')
        print(f'phi_1: {round(phase, 3)} ± {round(phase_std, 3)}')
        print(f'mean: {round(mean, 3)} ± {round(mean_std, 3)}')
        print(f'A_2: {round(amp2, 3)} ± {round(amp2_std, 3)}')
        print(f'phi_2: {round(phase2, 3)} ± {round(phase2_std, 3)}')
        return data_fit


    def plot_fit(self, ax, time, data_fit, **kwargs):
        ax.plot_date(time, data_fit, **kwargs)

    def plot_pol(self, ax, ant_i, chan_i):
        color, color2, color3 = ['y', 'c'], ['tab:green', 'tab:blue'], ['darkgreen', 'blue']
        id_i = f'{ant_i+1}{chan_i}'
        print(f'\n- Ant. {ant_i+1}, Pol. {chan_i+1}')

        average_rms = self.dataframe['average'+id_i] - np.mean(self.dataframe['average'+id_i]) #? center data
        time = self.dataframe['time']
        rms_cent = self.dataframe['rms'+id_i] - np.mean(self.dataframe['rms'+id_i])
        ax.plot_date(time, rms_cent, fmt=',', c=color[chan_i], alpha=0.5, label=f'Ant. {ant_i+1}, Pol. {chan_i}')
        ax.plot_date(time, average_rms, fmt=',', c=color2[chan_i], alpha=0.4, label=f'Moving avg Ant. {ant_i+1}, Ch. {chan_i+1}')

        data_fit = self.fit_sin(time, rms_cent)
        start_time, end_time = self.init_time, self.final_time
        mask = (time >= start_time) & (time <= end_time)
        time, rms_cent, data_fit, average_rms = time[mask], rms_cent[mask], data_fit[mask], average_rms[mask]

        self.plot_fit(ax, time, data_fit, lw=1, fmt='--', label=f'Fit rms Ant. {ant_i+1}, Ch. {chan_i+1}', c=color3[chan_i])
        mse = np.mean((average_rms - data_fit) ** 2)
        print(f'mse: {round(mse,3)}')


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

        plt.savefig('fit.png')
        plt.close()


filename = 'GalOscillation_Deconvolved_140-190.npz'
fit = fit_data(filename)
fit.process_data()
year1 = "2023"
year2 = "2023"
startTime = '{0}-05-05'.format(year1)
endTime = '{0}-05-15'.format(year2)
fit.set_window_time(startTime, endTime)
fit.plot_rms()