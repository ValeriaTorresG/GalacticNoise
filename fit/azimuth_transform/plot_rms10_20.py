from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from astropy import units as u
from astropy.time import Time
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np
import argparse
import logging
import astropy
import time
import re
import os

from astropy.visualization import astropy_mpl_style, quantity_support
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 150
plt.style.use(astropy_mpl_style)
quantity_support()

# python plot_rms10_20.py -f GalOscillation_Time_2024_01_05-2024_01_16_Freq_90-110.npz


class fit_data:

    def __init__(self, filename, runStatType, timeType, init_time, final_time, freqBand):
        """
        Initializes the fit_data class instance.

        Args:
            filename (str): Name of the .npz file containing data.
            runStatType (str): Type of statistic for moving average ('mean' or 'median').
            timeType (str): Type of time to use ('sid' or 'utc').
            init_time (str): Initial date in 'YYYY-MM-DD' format.
            final_time (str): Final date in 'YYYY-MM-DD' format.
            freqBand (str): Frequency band in MHz.
        """
        self.ant, self.pol = 3, 2
        self.id_list = [f'rms{_}{__}' for (_,__) in  list(product(np.arange(1,self.ant+1),np.arange(1,self.pol+1)))]
        self.id_spread = [f'spread_rms{_}{__}' for (_,__) in  list(product(np.arange(1,self.ant+1),np.arange(1,self.pol+1)))]
        self.window = 150
        self.timeType = timeType
        self.dataframe = pd.DataFrame()
        self.time, self.rms, self.spread = self.read_npz(filename)
        self.runStatType = runStatType
        self.init_time, self.final_time = init_time, final_time
        self.freqBand = freqBand
        logging.basicConfig(filename='fit_icecube.log', level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit_icecube')
        print(f'-> Reading {filename.split("_")[2][:-4]} MHz')


    def read_npz(self, filename):
        """
        Reads .npz files and returns time and RMS values.

        Args:
            filename (str): Name of the .npz file to read.

        Returns:
            tuple: time (array of times), rms (dictionary of RMS values), spread (dictionary of RMS spread values).
        """
        data = np.load(filename, allow_pickle=True)
        self.time = data['time']
        self.rms = dict(rms11 = data['rms10'], rms12 = data['rms11'],
                        rms21 = data['rms20'], rms22 = data['rms21'],
                        rms31 = data['rms30'], rms32 = data['rms31'])
        self.spread = dict(spread_rms11 = data['rms10_spread'], spread_rms12 = data['rms11_spread'],
                           spread_rms21 = data['rms20_spread'], spread_rms22 = data['rms21_spread'],
                           spread_rms31 = data['rms30_spread'], spread_rms32 = data['rms31_spread'])
        return self.time, self.rms, self.spread


    def to_dataframe(self, dataFrame=None):
        """
        Converts time and RMS values into a pandas DataFrame.

        Args:
            dataFrame (pd.DataFrame, optional): Existing DataFrame. If not provided, a new one is used.

        Returns:
            pd.DataFrame: DataFrame with time and RMS columns.
        """
        dataFrame['time'] = self.time
        for i, _ in enumerate(self.id_list):
            dataFrame[_] = self.rms[_]
            dataFrame[f'spread_{_}'] = self.spread[self.id_spread[i]]
        print("Converted to DataFrame")
        return dataFrame


    def selectTimePeriod(self, init_time, final_time, dataFrame=None):
        """
        Selects a specific time period from the DataFrame.

        Args:
            init_time (str): Initial date in 'YYYY-MM-DD' format.
            final_time (str): Final date in 'YYYY-MM-DD' format.
            dataFrame (pd.DataFrame, optional): DataFrame to filter.

        Returns:
            pd.DataFrame: DataFrame filtered by the specified time period.
        """
        init_time, final_time = datetime.strptime(init_time, "%Y-%m-%d"), datetime.strptime(final_time, "%Y-%m-%d")
        dataFrame = dataFrame.loc[dataFrame['time'] > init_time]
        dataFrame = dataFrame.loc[dataFrame['time'] < final_time]
        print(f'Selected time period from {init_time} to {final_time}')
        return dataFrame#.dropna()


    def clean_data(self, n=50, dataFrame=None):
        """
        Cleans data by removing outliers that exceed a threshold defined by 'n'.

        Args:
            n (int, optional): Threshold to remove outliers. Default is 50.
            dataFrame (pd.DataFrame, optional): DataFrame to clean.

        Returns:
            pd.DataFrame: DataFrame without outlier points.
        """
        for _ in self.id_list:
            mean_ = np.mean(dataFrame[_])
            dataFrame = dataFrame[np.abs(dataFrame[_]) - mean_ <=  n]
        print("Cleaned Outliers Points")
        return dataFrame


    def get_moving_avg(self, dataFrame):
        """
        Calculates the moving average using a rolling window for each day.

        Args:
            dataFrame (pd.DataFrame): DataFrame with data to process.

        Returns:
            pd.DataFrame: DataFrame with calculated moving average.
        """
        grouped = dataFrame.groupby(dataFrame['time'].dt.date)
        result_df = pd.DataFrame()
        for date, group in grouped:
            for _ in self.id_list:
                if self.runStatType == 'mean':
                    group['average'+_] = group[_].rolling(self.window, win_type="gaussian").mean(std=3)
                elif self.runStatType == 'median':
                    group['average'+_] = group[_].rolling(self.window).median()
                else:
                    raise ValueError('Invalid runStatType')
            result_df = pd.concat([result_df, group])
        print(f"Calculated Moving {self.runStatType}")
        return result_df


    def clean_outliers(self, dataFrame, threshold=5.0):
        """
        Removes outliers that are more than 'threshold' standard deviations from the mean.

        Args:
            dataFrame (pd.DataFrame): DataFrame to clean.
            threshold (float, optional): Number of standard deviations to consider as outlier. Default is 5.0.

        Returns:
            pd.DataFrame: DataFrame without outliers.
        """
        for _ in self.id_list:
            rms = dataFrame[_]
            mean = np.mean(rms)
            std = np.std(rms)
            dataFrame = dataFrame[(rms - mean) <= threshold * std]
        return dataFrame


    def process_data(self):
        """
        Processes the data by applying several cleaning and transformation functions.

        Returns:
            pd.DataFrame: Processed DataFrame ready for analysis and visualization.
        """
        df = self.to_dataframe(self.dataframe)
        df = self.selectTimePeriod(init_time=self.init_time, final_time=self.final_time, dataFrame=df)
        df = self.clean_data(n=10, dataFrame=df)
        df = self.clean_outliers(dataFrame=df)
        df = self.get_moving_avg(dataFrame=df)
        print("Finished Processing Data")
        return df


    def get_sidereal_time(self, time_utc):
        """
        Converts UTC time to local sidereal time.

        Args:
            time_utc (array): Array of times in UTC.

        Returns:
            array: Array of times in local sidereal time.
        """
        location = astropy.coordinates.EarthLocation.of_site("IceCube")
        time = astropy.time.Time(time_utc, location=location)
        sidereal_time = time.sidereal_time('apparent').value
        return sidereal_time


    def bin_data(self, time, rms, spread=None):
        """
        Bins the data using the median and removes empty or non-numeric bins to avoid errors in the fit.

        Args:
            time (array): Array of times (can be in sidereal time).
            rms (array): Array of corresponding RMS values.
            spread (array): Array of corresponding spread values.

        Returns:
            tuple: bins, bin_centers_, sem, bin_centers, bin_std, spread
        """
        bins, bin_edges, _ = binned_statistic(time, rms, statistic='median', bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        bin_std , _, _ = binned_statistic(time, rms, statistic='std', bins=50)
        bin_counts, _, _ = binned_statistic(time, rms, statistic='count', bins=50)
        sem = bin_std/np.sqrt(bin_counts)
        mask =  (~np.isnan(bins)) & (~np.isinf(bins))# & (bin_counts>1)

        if spread is not None:
            bins_sp, _, _ = binned_statistic(time, spread, statistic='median', bins=50)
            return bins[mask], bin_centers[mask], sem[mask], bin_centers, bin_std[mask], bins_sp[mask]

        return bins[mask], bin_centers[mask], sem[mask], bin_centers, bin_std[mask]


    def chi_squared(self, y_obs, y_fit, sigma):
        """
        Calculates the reduced chi-squared value to evaluate the fit.

        Args:
            y_obs (array): Observed values.
            y_fit (array): Fitted values.
            sigma (array): Standard deviations of the observed values.

        Returns:
            float: Reduced chi-squared value.
        """
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-3)


    def fit_sin(self, time, rms, ant_i, chan_i):
        """
        Fits a sinusoidal function to the RMS data as a function of sidereal time.

        Args:
            time (array): Array of times in UTC.
            rms (array): Array of RMS values.
            ant_i (int): Antenna number.
            chan_i (int): Channel (polarization) number.

        Returns:
            tuple: bin_centers_ (centers of the bins), data_fit (fitted values).
        """
        t = self.get_sidereal_time(time)

        bins, bin_centers, sem_vals, bin_centers_, std = self.bin_data(t, rms)

        guess_mean, guess_std = np.mean(bins),  np.std(bins)
        guess_phase, guess_freq = 1.0, 2/23.934470
        guess_amp = guess_std/np.sqrt(2)
        bin_centers = np.asarray(bin_centers)

        def optimize_func(x_bins, amp, phase, mean):
            return amp * np.sin(2 * np.pi * guess_freq * x_bins + phase) + mean

        popt, pcov = curve_fit(optimize_func, bin_centers, bins,
                               p0=[guess_amp, guess_phase, guess_mean],
                               bounds=([-np.inf,-np.inf, -np.inf],[np.inf, np.inf,np.inf]))
        est_amp, est_phase, est_mean = popt
        err_amp, err_phase, err_mean = np.sqrt(np.diag(pcov))
        data_fit = est_amp * np.sin(2 * np.pi * guess_freq * bin_centers_ + est_phase) + est_mean

        if est_amp<0:
            est_amp = np.abs(est_amp)
            est_phase = est_phase + np.pi
        print(f"Fitted parameters: Amp={est_amp}, Phase={np.rad2deg(est_phase)}, Mean={est_mean}")
        print(f"Parameter uncertainties: dAmp={err_amp},  dPhase={np.rad2deg(err_phase)}, dMean={err_mean}")
        self.logger.info({'ant':ant_i, 'pol':chan_i, 'init_time':self.init_time,
                            'final_time':self.final_time, 'Amp':est_amp,
                            'phase':est_phase, 'mean':est_mean, 'dAmp':err_amp,
                            'dPhase':err_phase, 'dMean':err_mean})

        return bin_centers_, data_fit


    def plot_pol(self, dataFrame, ax, ant_i, chan_i, fitUniqueDays=None, fig=None):
        """
        Plots the RMS values for a specific antenna and polarization.
        Plots a fit for each day if fitUniqueDays is True.
        Error bars use the std from the binning process.

        Args:
            dataFrame (pd.DataFrame): DataFrame with data to plot.
            ax (matplotlib.axes.Axes): Axis where the plot will be made.
            ant_i (int): Antenna number.
            chan_i (int): Channel (polarization) number.
            fitUniqueDays (bool, optional): If True, fits each day separately.
            cmap (ListedColormap, optional): Color map to use.
        """
        id_i = f'{ant_i}{chan_i}'
        print(f'\n- Ant. {ant_i}, Pol. {chan_i}')

        time = dataFrame['time']
        average_rms = dataFrame['averagerms'+id_i] - np.mean(dataFrame['averagerms'+id_i])  # center data
        spread = dataFrame['spread_rms'+id_i]
        rms_cent = dataFrame['rms'+id_i] - np.mean(dataFrame['rms'+id_i])

        unique_days = time.dt.date.unique()

        cmap = ListedColormap(sns.color_palette("Spectral", n_colors=256))
        norm = Normalize(vmin=spread.min(), vmax=spread.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for i, day in enumerate(unique_days):
            day_mask = pd.to_datetime(dataFrame['time']).dt.date == day
            time_i = time[day_mask]
            spread_i = spread[day_mask]
            day_rms_cent = rms_cent[day_mask]
            day_average_rms = average_rms[day_mask]
            if len(day_rms_cent) == 0:
                print(f"No data for day {day}, skipping.")
            else:
                if fitUniqueDays == 'True':
                    print(f'Fitting unique day {day}')
                    x_interpolated, y_interpolated = self.fit_sin(time_i, day_average_rms, ant_i, chan_i)
                else:
                    
                    bins, bin_centers, sem_vals, _, _, spread_i = self.bin_data(self.get_sidereal_time(time_i), day_average_rms, spread=spread_i)
                    color = cmap(norm(spread_i))
                    ax.errorbar(bin_centers, bins, yerr=sem_vals, markersize=3.0, fmt='o', ecolor=color, markeredgecolor=color, markerfacecolor=color, alpha=0.6)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                                orientation='vertical', pad=0.04)
                    cbar.set_label('RMS spread')

        #x_interpolated, y_interpolated = self.fit_sin(time, average_rms, ant_i, chan_i)
        #bins, bin_centers, _, _, std = self.bin_data(self.get_sidereal_time(time), average_rms)
        #ax.errorbar(bin_centers, bins, yerr=std, markersize=5.0, elinewidth=1.0, capsize=2, capthick=1.2,
                    #fmt='o', ecolor='black', markeredgecolor='black', markerfacecolor='black')
        #ax.plot(x_interpolated, y_interpolated, lw=2.0, ls='--', label=f'Fit rms Ant. {ant_i}, Ch. {chan_i}', c='black')


    def plot_rms(self, dataFrame, plotAll=True, ant_ch_List=[[1,1],[1,2]], fitUniqueDays=False):
        """
        Generates plots of the RMS values for specified antennas and polarizations.

        Args:
            dataFrame (pd.DataFrame): DataFrame with data to plot.
            plotAll (bool, optional): If True, plots all available antennas and polarizations. Default is True.
            ant_ch_List (list, optional): List of antennas and polarizations to plot if plotAll is False.
            fitUniqueDays (bool, optional): If True, fits each day separately.
        """
        if not plotAll:
            assert len(ant_ch_List) > 0, "ant_ch_List must have at least one element"
            plotList = ant_ch_List
            print(f'Plotting only antennas and polarisations: {plotList}')
        else:
            plotList = list(product(np.arange(1,3+1), np.arange(1,2+1)))  # Not using self, incase loaded from a file
            print(f'Plotting all antennas and polarisations: {plotList}')
        unqAntID = np.unique(np.array(plotList)[:,0])
        #=== Initiate a Figure ===
        numRows = len(unqAntID)
        fig, axs = plt.subplots(numRows, 2, figsize=[25, 16])

        for ant_i in unqAntID:
            corresChans = [_[1] for _ in plotList if _[0] == ant_i]
            for chan_i in corresChans:
                self.plot_pol(dataFrame, axs[ant_i-1, chan_i-1], ant_i, chan_i, fitUniqueDays, fig)

        for i, ax in enumerate(axs.ravel()):
            ax.set_ylabel('RMS')#; ax.set_ylim(-2,2)
            #ax.set_xlabel('Azimuth [deg]')
            ax.set_xlabel('Sidereal Time [h]')
            ax.legend(fontsize=15)
            ax.grid(True)
            #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norms[i]), ax=ax,
                                #orientation='vertical', pad=0.04)
            #cbar.set_label('RMS spread')

        plt.suptitle(f'Galactic Noise\nFrequency band: {self.freqBand} MHz\n From {self.init_time} to {self.final_time}',
                     fontsize=18, y=0.95)
        plt.savefig(f'plot.png', dpi=360)
        plt.close()
        self.logger.info('-')


parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='.npz file name', required=True)
parser.add_argument('--stat', '-st', type=str, help='mean or median', required=False, default='mean')
parser.add_argument('--init_time', '-ini', type=str, help='initial time -> YYYY-MM-DD', required=False)
parser.add_argument('--final_time', '-fin', type=str, help='final time -> YYYY-MM-DD', required=False)
parser.add_argument('--use_time', '-t', type=str, help='sid or utc', required=False, default='sid')
parser.add_argument('--fitUniqDays', '-fUD', type=str, help='Fit unique days', required=False, default=False)
args = parser.parse_args()

if not args.init_time or not args.final_time:
    match = re.search(r'(\d{4}_\d{2}_\d{2})-(\d{4}_\d{2}_\d{2})', args.file)
    init_time = match.group(1).replace('_', '-')
    final_time = match.group(2).replace('_', '-')
else:
    init_time = args.init_time
    final_time = args.final_time

init = time.time()
baseLoc  = '/data/user/valeriatorres/galactic_noise/SouthPole/'
fileLoc = os.path.join(baseLoc, args.file)

fit = fit_data(filename = fileLoc, runStatType = args.stat,
               timeType = args.use_time, init_time = init_time,
               final_time = final_time, freqBand = args.file.split('_')[-1][:-4])
df = fit.process_data()
#df = pd.read_hdf(f'/data/user/valeriatorres/galactic_noise/SouthPole/dfs/{None}_{None}.h5')
#df.to_hdf(f'/data/user/valeriatorres/galactic_noise/SouthPole/dfs/{args.init_time}_{args.final_time}.h5', key='df', mode='w')

#print("=== Plotting ===")
fit.plot_rms(dataFrame=df, plotAll=True, ant_ch_List=[[1,1],[1,2],[2,1]], fitUniqueDays=args.fitUniqDays)
print(f'{"-"*10}> time elapsed: {(time.time()-init)/60:.3f} min')