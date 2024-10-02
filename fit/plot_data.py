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

# python fit_sidereal.py -ini 2023-08-05 -fin 2023-09-15 -freq

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
        for _ in self.id_list:
          if self.runStatType == 'mean':
            dataFrame['average'+_] = dataFrame[_].rolling(self.window, win_type="gaussian").mean(std=3)
          elif self.runStatType == 'median':
            dataFrame['average'+_] = dataFrame[_].rolling(self.window).median()
          else:
            raise ValueError('Invalid runStatType')
        print(f"Calculated Moving {self.runStatType}")
        return dataFrame

    def clean_outliers(self, dataFrame, threshold=5.0):  # remove outliers that are more than n standard deviations from the mean
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
        #df = self.filter_signal(dataFrame=df)
        print("Finished Processing Data")
        return df

    def plot_pol(self, dataFrame, ax, ant_i,chan_i, fitUniqueDays=None):
        id_i = f'{ant_i}{chan_i}'
        print(f'\n- Ant. {ant_i}, Pol. {chan_i}')

        time = dataFrame['time']
        average_rms = dataFrame['averagerms'+id_i] - np.mean(dataFrame['averagerms'+id_i])  # center data
        rms_cent = dataFrame['rms'+id_i] - np.mean(dataFrame['rms'+id_i])

        unique_days = pd.to_datetime(dataFrame['time']).dt.date.unique()
        cmap = ListedColormap(sns.color_palette("Spectral", as_cmap=True)(np.linspace(0., 1., len(unique_days))))

        for i, day in enumerate(unique_days):
            day_mask = pd.to_datetime(dataFrame['time']).dt.date == day
            time_i = time[day_mask]
            day_rms_cent = rms_cent[day_mask]
            day_average_rms = average_rms[day_mask]
            color = cmap(i / (len(unique_days)))
            ax.plot_date(time_i, day_average_rms, color=color, ms=1.0)
            #ax.set_xticklabels(ax.get_xticks(), rotation=40)

    def plot_rms(self, dataFrame,  month, plotAll=True, ant_ch_List=[[1,1],[1,2]], fitUniqueDays=False):
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
          ax.set_ylabel('RMS')#; ax.set_ylim(-2,2)
          #ax.set_xlabel('Sidereal time')
          #ax.legend(fontsize=15)

      plt.suptitle(f'Galactic Noise\nFrequency band: {self.freqBand} MHz\n{month} 2023', fontsize=18, y=0.95)
      plt.savefig(f'{month}.png', dpi=360)
      plt.close()
      self.logger.info('-')


parser = argparse.ArgumentParser()
parser.add_argument('--stat', '-st', type=str, help='mean or median', required=False, default='mean')
#parser.add_argument('--init_time', '-ini', type=str, help='initial time -> YYYY-MM-DD', required=True)
#parser.add_argument('--final_time', '-fin', type=str, help='final time -> YYYY-MM-DD', required=True)
parser.add_argument('--freq_band', '-freq', type=str, help='', required=False, default='70-150')
parser.add_argument('--use_time', '-t', type=str, help='sid or utc', required=False, default='sid')
parser.add_argument('--fitUniqDays', '-fUD', type=str, help='Fit unique days', required=False, default=False)
args = parser.parse_args()

#init_time = args.init_time
#final_time = args.final_time

init = time.time()
baseLoc = '/data/user/valeriatorres/galactic_noise/SouthPole/daily/'
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
i_dates = ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01','2023-06-01',
         '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', '2024-01-01']
f_dates = ['2023-01-31', '2023-02-28', '2023-03-31', '2023-04-30', '2023-05-31','2023-06-30',
         '2023-07-31', '2023-08-31', '2023-09-30', '2023-10-31', '2023-11-30', '2023-12-31']
for i in range(len(i_dates)-1):
    print(i_dates[i])
    month = months[i]
    fit = fit_data(base_path=baseLoc, runStatType=args.stat, timeType=args.use_time, init_time=i_dates[i], final_time=i_dates[i+1], freqBand=args.freq_band)
    df = fit.process_data()
    fit.plot_rms(dataFrame=df, month=month, plotAll=True, ant_ch_List=[[1,1],[1,2],[2,1]], fitUniqueDays=args.fitUniqDays)
print(f'{"-"*10}> time elapsed: {(time.time()-init)/60:.3f} min')