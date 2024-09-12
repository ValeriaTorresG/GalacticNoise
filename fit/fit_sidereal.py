from scipy.stats import binned_statistic
from scipy.optimize import curve_fit, leastsq, least_squares
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
plt.rcParams['figure.dpi'] = 150
plt.style.use(astropy_mpl_style)
quantity_support()

# python fit_sidereal.py -f GalOscillation_Time_2024_01_05-2024_01_26_Freq_70-150.npz -ini 2024-01-05 -fin 2024-01-26 -fUD False

class fit_data:

    def __init__(self, filename, runStatType, timeType,
                 init_time, final_time, freqBand):

        self.ant, self.pol = 3, 2 #* 3 antennas and 2 polarisations

        self.id_list = [f'rms{_}{__}' for (_,__) in  list(product(np.arange(1,self.ant+1),np.arange(1,self.pol+1)))]

        self.window = 150
        self.timeType = timeType
        self.dataframe = pd.DataFrame() #* create dataframe to make easier column operations
        self.time, self.rms = self.read_npz(filename) #* read compressed numpy files

        self.runStatType = runStatType

        self.init_time, self.final_time = init_time, final_time
        self.freqBand = freqBand

        logging.basicConfig(filename='fit_icecube.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger('fit_icecube')
        print(f'-> Reading {filename.split("_")[2][:-4]} MHz')


    def read_npz(self, filename):
        data = np.load(filename, allow_pickle=True) #* allow pickling -> use of packed objects
        self.time = data['time']
        self.rms = dict(
                      rms11 = data['rms10'], rms12 = data['rms11'],
                      rms21 = data['rms20'], rms22 = data['rms21'],
                      rms31 = data['rms30'], rms32 = data['rms31']
                          )
        return self.time, self.rms

    def to_dataframe(self, dataFrame=None):
        dataFrame['time'] = self.time
        for _ in self.id_list:
            dataFrame[_] = self.rms[_]
        print("Converted to DataFrame")
        return dataFrame

    def selectTimePeriod(self, init_time, final_time, dataFrame=None):
        init_time, final_time = datetime.strptime(init_time, "%Y-%m-%d"), datetime.strptime(final_time, "%Y-%m-%d")
        dataFrame = dataFrame.loc[dataFrame['time'] > init_time]
        dataFrame = dataFrame.loc[dataFrame['time'] < final_time]
        print(f'Selected time period from {init_time} to {final_time}')
        return dataFrame

    def clean_data(self, n=50, dataFrame=None): # n to calc a threshold by adding n to the mean rms.
      for _ in self.id_list:
          mean_ = np.mean(dataFrame[_])
          dataFrame = dataFrame[np.abs(dataFrame[_]) - mean_ <=  n] #* remove outliers
      print("Cleaned Outliers Points")
      return dataFrame


    def get_moving_avg(self, dataFrame): #* calc moving avg using rolling
        for _ in self.id_list:
          if self.runStatType == 'mean':
            dataFrame['average'+_] = dataFrame[_].rolling(self.window, win_type="gaussian").mean(std=3)
          elif self.runStatType == 'median':
            dataFrame['average'+_] = dataFrame[_].rolling(self.window).median()
          else:
            raise ValueError('Invalid runStatType')
        print(f"Calculated Moving {self.runStatType}")
        return dataFrame


    def clean_outliers(self, dataFrame, threshold=5.0): # remove outliers that are more than n standard deviations from the mean
        for _ in self.id_list:
          rms = dataFrame[_]
          mean = np.mean(rms)
          std = np.std(rms)
          dataFrame = dataFrame[(rms - mean) <= threshold * std]
        return dataFrame

    def process_data(self):
        df = self.to_dataframe(self.dataframe)
        df = self.selectTimePeriod(init_time=self.init_time, final_time=self.final_time, dataFrame=df)
        df = self.clean_data(n=10, dataFrame=df)
        df = self.get_moving_avg(dataFrame=df)
        df = self.clean_outliers(dataFrame=df)
        print("Finished Processing Data")
        return df

    #======= Fit and Plotting Functions =======

    def get_sidereal_time(self, time_utc):
        location = astropy.coordinates.EarthLocation.of_site("IceCube")
        time = astropy.time.Time(time_utc, location=location)
        sidereal_time = time.sidereal_time('apparent').value
        return sidereal_time


    def bin_data(self, time, rms):
        bins, bin_edges, _ = binned_statistic(time, rms, statistic='median', bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        bin_std , _, _ = binned_statistic(time, rms, statistic='std', bins=50)
        bin_counts, _, _ = binned_statistic(time, rms, statistic='count', bins=50)
        sem = bin_std/np.sqrt(bin_counts)
        mask =  ~np.isnan(bins) & ~np.isinf(bins)
        bin_centers, bins, sem = bin_centers[mask], bins[mask], sem[mask]

        print(f'Number of bins: {len(bins)}')
        return bins, bin_centers, sem


    def chi_squared(self, y_obs, y_fit, sigma):
        return np.sum(((y_fit-y_obs)/sigma)**2)/(len(y_obs)-3)


    def fit_sin(self, time, rms, ant_i, chan_i):

      t = self.get_sidereal_time(time)

      bins, bin_centers, sem_vals = self.bin_data(t, rms)

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

      data_fit = est_amp * np.sin(2 * np.pi * guess_freq * bin_centers + est_phase) + est_mean
      self.logger.info({'ant':ant_i, 'pol':chan_i, 'init_time':self.init_time, 'final_time':self.final_time,
                        'Amp':est_amp, 'phase':est_phase, 'mean':est_mean,
                        'dAmp':err_amp, 'dPhase':err_phase, 'dMean':err_mean})

      return bin_centers, data_fit


    def plot_pol(self,dataFrame, ax, ant_i,chan_i, fitUniqueDays=None):
        id_i = f'{ant_i}{chan_i}'
        print(f'\n- Ant. {ant_i}, Pol. {chan_i}')

        time = dataFrame['time']
        average_rms = dataFrame['averagerms'+id_i] - np.mean(dataFrame['averagerms'+id_i]) #center data
        rms_cent = dataFrame['rms'+id_i] - np.mean(dataFrame['rms'+id_i])

        sidereal_times = np.array([self.get_sidereal_time(t) for t in time])
        unique_days = pd.to_datetime(dataFrame['time']).dt.date.unique()

        # Define a colormap with 5 discrete colors
        cmap = ListedColormap(sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.2, 0.7, len(unique_days))))

        for i, day in enumerate(unique_days):
            day_mask = pd.to_datetime(dataFrame['time']).dt.date == day
            day_sidereal_times = sidereal_times[day_mask]
            time_i = time[day_mask]
            day_rms_cent = rms_cent[day_mask]
            day_average_rms = average_rms[day_mask]
            # ax.scatter(day_sidereal_times, day_rms_cent, alpha=0.7, s=0.6, color=colors[i])
            # ax.scatter(day_sidereal_times, day_average_rms, color=cmap(i / (len(unique_days)-1)), s=1.5, alpha=0.4)
            color = cmap(i / (len(unique_days)-1))
            if fitUniqueDays=='True':
              print(f'Fitting unique day {day}')
              x_interpolated, y_interpolated = self.fit_sin(time_i, day_average_rms, ant_i, chan_i)
              # ax.plot(x_interpolated, y_interpolated, lw=2.0, label=f'{day}', color=cmap(i / (len(unique_days)-1)))
            else:
              # Only plot the average
              bins, bin_centers, sem_vals = self.bin_data(self.get_sidereal_time(time_i), day_average_rms)
              ax.errorbar(bin_centers, bins, yerr=sem_vals, markersize=4.0,
                          fmt ='o', ecolor=color,
                          markeredgecolor=color, markerfacecolor=color,alpha=0.8)
              #plt.savefig(f'plot_{ant_i}_{chan_i}.png', dpi=360)

        x_interpolated, y_interpolated = self.fit_sin(time, average_rms, ant_i, chan_i)
        bins, bin_centers, sem_vals = self.bin_data(self.get_sidereal_time(time), average_rms)
        ax.errorbar(bin_centers, bins, yerr=sem_vals, markersize=7.0, fmt ='o', ecolor='black', markeredgecolor='black', markerfacecolor='black')
        ax.plot(x_interpolated, y_interpolated, lw=2.0, ls='--', label=f'Fit rms Ant. {ant_i}, Ch. {chan_i}', c='black')

        x = self.get_sidereal_time(time)
        y = self.bin_data(x, average_rms)
        #print(f'mse: {np.mean((y-y_interpolated)**2)}')
        #print(f'Chi-squared: {self.chi_squared(y, y_interpolated, np.std(y_interpolated)):.3f}')


    def plot_rms(self, dataFrame, plotAll=True, ant_ch_List=[[1,1],[1,2]], fitUniqueDays=False):
      if not plotAll:
        assert len(ant_ch_List) > 0, "ant_ch_List must have at least one element"
        plotList = ant_ch_List
        print(f'Plotting only antennas and polarisations: {plotList}')
      else:
        plotList = list(product(np.arange(1,3+1), np.arange(1,2+1))) # Not using self, incase loaded from a file
        print(f'Plotting all antennas and polarisations: {plotList}')
      unqAntID = np.unique(np.array(plotList)[:,0])
      #=== Initiate a Figure ===
      numRows = len(unqAntID)
      fig, axs = plt.subplots(numRows,2, figsize=[25, 16])
      # if len(unqAntID) > 1:axs = axs.ravel()
      for ant_i in unqAntID:
          corresChans = [_[1] for _ in plotList if _[0] == ant_i]
          for chan_i in corresChans:
            self.plot_pol(dataFrame, axs[ant_i-1,chan_i-1], ant_i, chan_i,fitUniqueDays)

      # Set Y lims for all plots
      for ax in axs.ravel():
          ax.set_ylabel('RMS'); ax.set_ylim(-2,2)
          ax.set_xlabel('Sidereal time')
          ax.legend(fontsize=15)

      plt.suptitle(f'Galactic Noise\nFrequency band: {self.freqBand} MHz\n From {self.init_time} to {self.final_time}', fontsize=18, y=1.01)
      plt.savefig(f'../plots/plots_icecube/fit_{self.init_time}_{self.final_time}_70-150mhz.png', dpi=360)
      plt.close()
      self.logger.info('-')

    def do_MCMC():
        two_pi = 2 * np.pi
        pass #TODO implement MCMC

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, help='.npz file name', required=True)
parser.add_argument('--stat', '-st', type=str, help='mean or median', required=False, default='mean')
parser.add_argument('--init_time', '-ini', type=str, help='initial time -> YYYY-MM-DD', required=False)
parser.add_argument('--final_time', '-fin', type=str, help='final time -> YYYY-MM-DD', required=False)
parser.add_argument('--use_time', '-t', type=str, help='sid or utc', required=False, default='sid')
parser.add_argument('--fitUniqDays', '-fUD', type=str, help='Fit unique days', required=True)
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

#==== The Magic Happens Here ====
fit = fit_data(filename = fileLoc, runStatType = args.stat,
               timeType = args.use_time, init_time=init_time,
               final_time=final_time, freqBand = args.file.split('_')[-1][:-4])
df = fit.process_data()
# For Bigger Files Maybe you want to save the processed data
#df.to_hdf(f'/data/user/valeriatorres/galactic_noise/SouthPole/byParas/{args.init_time}_{args.final_time}.h5', key='df', mode='w')

#=== Plotting ===
print("=== Plotting ===")
fit.plot_rms(dataFrame=df, plotAll=True, ant_ch_List=[[1,1],[1,2],[2,1]], fitUniqueDays=args.fitUniqDays)
print(f'{"-"*10}> time elapsed: {(time.time()-init)/60:.3f} min')