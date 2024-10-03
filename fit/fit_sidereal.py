from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from astropy import units as u
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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 360
plt.style.use(astropy_mpl_style)
quantity_support()

# python fit_sidereal.py -ini 2023-08-05 -fin 2023-08-15

class fit_data:

    def __init__(self, base_path, runStatType, timeType, init_time, final_time, freqBand):

        self.ant, self.pol = 3, 2  # * 3 antennas and 2 polarisations
        self.id_list = [f'rms{_}{__}' for (_,__) in list(product(np.arange(1,self.ant+1), np.arange(1,self.pol+1)))]

        self.window = 150
        self.timeType = timeType
        self.dataframe = pd.DataFrame()  # * create dataframe to make easier column operations
        self.runStatType = runStatType

        self.init_time, self.final_time = init_time, final_time
        self.freqBand = freqBand

        self.base_path = base_path
        self.files = self.get_files_by_date(init_time, final_time)

        logging.basicConfig(filename='fit_icecube.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit_icecube')
        #print(f'-> Found {len(self.files)} files between {init_time} and {final_time}')


    def get_files_by_date(self, init_time, final_time):
        # Convert dates to datetime objects
        init_time = datetime.strptime(init_time, "%Y-%m-%d")
        final_time = datetime.strptime(final_time, "%Y-%m-%d")

        # List files in the directory
        files = os.listdir(self.base_path)

        # Filter files matching the format and within the date range
        pattern = r"GalOscillation_Time_(\d{4})-(\d{2})-(\d{2})_Freq_"+ self.freqBand + r"\.npz"
        valid_files = []
        for file in files:
            match = re.match(pattern, file)
            if match:
                file_date = datetime(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))
                if init_time <= file_date <= final_time:
                    valid_files.append(os.path.join(self.base_path, file))
        print(f'{len(valid_files)} days of data')
        return valid_files

    def read_npz(self, filename):
        data = np.load(filename, allow_pickle=True)  # * allow pickling -> use of packed objects
        self.time = data['time']
        self.rms = dict(
                      rms11 = data['rms10'], rms12 = data['rms11'],
                      rms21 = data['rms20'], rms22 = data['rms21'],
                      rms31 = data['rms30'], rms32 = data['rms31']
                          )
        return self.time, self.rms

    def to_dataframe(self, dataFrame=None):
        for file in self.files:
            #print(f"-> Reading file {file}")
            self.time, self.rms = self.read_npz(file)
            temp_df = pd.DataFrame({'time': self.time})
            for _ in self.id_list:
                temp_df[_] = self.rms[_]
            self.dataframe = pd.concat([self.dataframe, temp_df])
        print("Converted to DataFrame")
        return self.dataframe

    def selectTimePeriod(self, init_time, final_time, dataFrame=None):
        assert 'time' in self.dataframe.columns, "There is no data for this period of time"
        init_time, final_time = datetime.strptime(init_time, "%Y-%m-%d"), datetime.strptime(final_time, "%Y-%m-%d")
        self.dataframe['time'] = pd.to_datetime(self.dataframe['time'], unit='s')
        self.dataframe = self.dataframe[(self.dataframe['time'] >= init_time) & (self.dataframe['time'] <= final_time)]
        print(f'Selected time period from {init_time} to {final_time}')
        return self.dataframe

    def clean_data(self, n=10, dataFrame=None):  # n to calculate a threshold by adding n to the mean rms.
      for _ in self.id_list:
          mean_ = np.mean(dataFrame[_])
          dataFrame = dataFrame[np.abs(dataFrame[_]) - mean_ <=  n]  # * remove outliers
      print("Cleaned Outliers Points")
      return dataFrame


    def get_moving_avg(self, dataFrame):  # * calc moving avg using rolling
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

    def clean_outliers(self, dataFrame, threshold=5.0):  # remove outliers that are more than n standard deviations from the mean
        for _ in self.id_list:
          rms = dataFrame[_]
          mean = np.mean(rms)
          std = np.std(rms)
          dataFrame = dataFrame[(rms - mean) <= threshold * std]
        return dataFrame

    def filter_signal(self, dataFrame, threshold=0.1):
        time = self.get_sidereal_time(dataFrame['time'])
        freq = np.fft.fftfreq(len(time), d=(time[1] - time[0]))
        mask = np.abs(freq) < 0.1
        for _ in self.id_list:
          rms = dataFrame[_].to_numpy().ravel()
          ft = fft(rms)
          ft_filt = ft * mask
          filtered_signal = ifft(ft_filt)
          dataFrame[_] = filtered_signal
        return dataFrame

    def process_data(self):
        df = self.to_dataframe(self.dataframe)
        df = self.selectTimePeriod(init_time=self.init_time, final_time=self.final_time, dataFrame=df)
        df = self.clean_data(n=10, dataFrame=df)
        df = self.get_moving_avg(dataFrame=df)
        df = self.clean_outliers(dataFrame=df)
        #df = self.filter_signal(dataFrame=df)
        print("Finished Processing Data")
        return df

    #======= Fit and Plotting Functions =======

    def get_sidereal_time(self, time_utc):
        location = astropy.coordinates.EarthLocation.of_site("IceCube")
        time = astropy.time.Time(time_utc, location=location)
        sidereal_time = time.sidereal_time('apparent').value
        
        return sidereal_time

    def get_azimuth(self, time):
        azimuth = (90 + time) % 360
        return azimuth

    def bin_data(self, time, rms):
        bins, bin_edges, _ = binned_statistic(time, rms, statistic='median', bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        bin_std , _, _ = binned_statistic(time, rms, statistic='std', bins=50)
        bin_counts, _, _ = binned_statistic(time, rms, statistic='count', bins=50)
        sem = bin_std/np.sqrt(bin_counts)
        mask =  ~np.isnan(bins) & ~np.isinf(bins)
        bin_centers_, bins, sem = bin_centers[mask], bins[mask], sem[mask]
        #print(f'Number of bins: {len(bins)}')
        return bins, bin_centers_, sem, bin_centers

    def chi_squared(self, y_obs, y_fit, sigma):
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-3)

    def fit_sin(self, time, rms, ant_i, chan_i):
      t = self.get_sidereal_time(time)
      azim = self.get_azimuth(t)

      #bins, bin_centers, sem_vals, bin_centers_ = self.bin_data(t, rms)
      bins, bin_centers, sem_vals, bin_centers_ = self.bin_data(azim, rms)

      guess_mean, guess_std = np.mean(bins),  np.std(bins)
      guess_phase, guess_freq = 1.0, 2/23.934470
      guess_amp = guess_std/np.sqrt(2)
      bin_centers = np.asarray(bin_centers)

      def optimize_func(x_bins, amp, phase, mean):
        return amp * np.sin(2 * np.pi * guess_freq * x_bins + phase) + mean

      popt, pcov = curve_fit(optimize_func, bin_centers, bins, p0=[guess_amp, guess_phase, guess_mean], bounds=([-np.inf,-np.inf, -np.inf],[np.inf, np.inf,np.inf]))
      est_amp, est_phase, est_mean = popt
      err_amp, err_phase, err_mean = np.sqrt(np.diag(pcov))

      # Print the results
      print(f"Fitted parameters: Amp={est_amp}, Phase={np.rad2deg(est_phase)}, Mean={est_mean}")
      print(f"Parameter uncertainties: dAmp={err_amp},  dPhase={np.rad2deg(err_phase)}, dMean={err_mean}")

      data_fit = est_amp * np.sin(2 * np.pi * guess_freq * bin_centers_ + est_phase) + est_mean
      self.logger.info({'ant':ant_i, 'pol':chan_i, 'init_time':self.init_time, 'final_time':self.final_time,
                        'Amp':est_amp, 'phase':est_phase, 'mean':est_mean,
                        'dAmp':err_amp, 'dPhase':err_phase, 'dMean':err_mean})

      return bin_centers_, data_fit

    def plot_pol(self, dataFrame, ax, ant_i,chan_i, fitUniqueDays=None):
        id_i = f'{ant_i}{chan_i}'
        print(f'\n- Ant. {ant_i}, Pol. {chan_i}')

        time = dataFrame['time']
        average_rms = dataFrame['averagerms'+id_i] - np.mean(dataFrame['averagerms'+id_i])  # center data
        rms_cent = dataFrame['rms'+id_i] - np.mean(dataFrame['rms'+id_i])

        sidereal_times = np.array([self.get_sidereal_time(t) for t in time])
        unique_days = pd.to_datetime(dataFrame['time']).dt.date.unique()

        cmap = ListedColormap(sns.color_palette("Spectral", as_cmap=True)(np.linspace(0., 1., len(unique_days))))

        for i, day in enumerate(unique_days):
            day_mask = pd.to_datetime(dataFrame['time']).dt.date == day
            day_sidereal_times = sidereal_times[day_mask]
            time_i = time[day_mask]
            day_rms_cent = rms_cent[day_mask]
            day_average_rms = average_rms[day_mask]
            color = cmap(i / (len(unique_days)))
            if fitUniqueDays == 'True':
                print(f'Fitting unique day {day}')
                x_interpolated, y_interpolated = self.fit_sin(time_i, day_average_rms, ant_i, chan_i)
            else:
                bins, bin_centers, sem_vals, _ = self.bin_data(self.get_azimuth(self.get_sidereal_time(time_i)), day_average_rms)
                ax.errorbar(bin_centers, bins, yerr=sem_vals, markersize=4.0,
                            fmt ='o', ecolor=color,
                            markeredgecolor=color, markerfacecolor=color,alpha=0.7)

        x_interpolated, y_interpolated = self.fit_sin(time, average_rms, ant_i, chan_i)
        bins, bin_centers, sem_vals, _ = self.bin_data(self.get_azimuth(self.get_sidereal_time(time)), average_rms)
        ax.errorbar(bin_centers, bins, yerr=np.std(bins), markersize=5.0, fmt ='o', ecolor='black', markeredgecolor='black', markerfacecolor='black')
        ax.plot(x_interpolated, y_interpolated, lw=2.0, ls='--', label=f'Fit rms Ant. {ant_i}, Ch. {chan_i}', c='black')

    def plot_rms(self, dataFrame, plotAll=True, ant_ch_List=[[1,1],[1,2]], fitUniqueDays=False):
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
      fig, axs = plt.subplots(numRows,2, figsize=[25, 16])
      for ant_i in unqAntID:
          corresChans = [_[1] for _ in plotList if _[0] == ant_i]
          for chan_i in corresChans:
            self.plot_pol(dataFrame, axs[ant_i-1,chan_i-1], ant_i, chan_i,fitUniqueDays)

      for ax in axs.ravel():
          ax.set_ylabel('RMS'); ax.set_ylim(-2,2)
          #ax.set_xlabel('Sidereal time')
          ax.set_xlabel('Azimuth [deg]')
          ax.legend(fontsize=15)

      plt.suptitle(f'Galactic Noise\nFrequency band: {self.freqBand} MHz\n From {self.init_time} to {self.final_time}', fontsize=18, y=0.95)
      plt.savefig(f'../plots/plots_icecube/fit_{self.init_time}_{self.final_time}_{self.freqBand}mhz.png', dpi=360)
      plt.close()
      self.logger.info('-')


parser = argparse.ArgumentParser()
parser.add_argument('--stat', '-st', type=str, help='mean or median', required=False, default='mean')
parser.add_argument('--init_time', '-ini', type=str, help='initial time -> YYYY-MM-DD', required=True)
parser.add_argument('--final_time', '-fin', type=str, help='final time -> YYYY-MM-DD', required=True)
parser.add_argument('--freq_band', '-freq', type=str, help='', required=False, default='70-150')
parser.add_argument('--use_time', '-t', type=str, help='sid or utc', required=False, default='sid')
parser.add_argument('--fitUniqDays', '-fUD', type=str, help='Fit unique days', required=False, default=False)
args = parser.parse_args()

init_time = args.init_time
final_time = args.final_time

init = time.time()
baseLoc = '/data/user/valeriatorres/galactic_noise/SouthPole/daily/'

#==== The Magic Happens Here ====
fit = fit_data(base_path=baseLoc, runStatType=args.stat, timeType=args.use_time, init_time=init_time, final_time=final_time, freqBand=args.freq_band)
df = fit.process_data()

print("=== Plotting ===")
fit.plot_rms(dataFrame=df, plotAll=True, ant_ch_List=[[1,1],[1,2],[2,1]], fitUniqueDays=args.fitUniqDays)
print(f'{"-"*10}> time elapsed: {(time.time()-init)/60:.3f} min')