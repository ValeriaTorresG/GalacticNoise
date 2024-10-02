from icecube.icetray import I3Units
from icecube.taxi_reader import taxi_tools
from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, taxi_reader, radcube
from icecube.icetray.i3logging import log_info
import matplotlib.gridspec as gridspec
from datetime import datetime
import time
import re
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse


#Input i3 file with the data
def get_i3_files(base_path='/mnt/ceph1-npx/user/valeriatorres/galactic_noise/SouthPole/i3_files', init='processed_'):
    files_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith(init):
                files_list.append(file)
    return files_list, base_path

def get_files_by_date(init_time, final_time):
        # Convert dates to datetime objects
        init_time = datetime.strptime(init_time, "%Y-%m-%d")
        final_time = datetime.strptime(final_time, "%Y-%m-%d")

        # List files in the directory
        files, base_path = get_i3_files()

        # Filter files matching the format and within the date range
        pattern = r"processed_(\d{4})-(\d{2})-(\d{2}).i3.gz"
        valid_files = []
        for file in files:
            match = re.match(pattern, file)
            if match:
                file_date = datetime(year=int(match.group(1)), month=int(match.group(2)), day=int(match.group(3)))
                if init_time <= file_date <= final_time:
                    valid_files.append(os.path.join(base_path, file))
        print(f'{len(valid_files)} days of data')
        return valid_files

filename = get_files_by_date('2023-08-05', '2023-09-15')

#Input i3 file with the data

#Set the trace length
WaveformLengths = [1024]

# This class take the spectrum average over all the Q frames
class AnalyzeQframes(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.NEntries = 0
        self.NFreqBins = int(WaveformLengths[0]/ 2 + 1)
        self.plotdBm = np.zeros(self.NFreqBins)
        self.plotFreqs = np.zeros(self.NFreqBins)
        self.channel_averages = {}  # Dictionary to store the average spectrum for each channel

    def SpectrumAverage(self, frame, name):
        if frame.Has('ArtifactsRemoved'):
            antennaDataMap = frame[name]  # This is a container that holds all antenna info
            for iant, antkey in enumerate(antennaDataMap.keys()):
                channelMap = antennaDataMap[antkey]
                for ichan, chkey in enumerate(channelMap.keys()):
                    fft = channelMap[chkey].GetFFTData()  # This container holds time series and spectrum
                    spectrum = fft.GetFrequencySpectrum()
                    freqs, amps = radcube.RadTraceToPythonList(spectrum)
                    amps = [radcube.GetDbmHzFromFourierAmplitude(abs(thisAmp), spectrum.binning, 50 * I3Units.ohm) for thisAmp in amps]

                    # Use (iant, ichan) as the key to differentiate data for each antenna and channel
                    if (iant, ichan) not in self.channel_averages:
                    # Create separate lists for each antenna and channel if not present
                        self.channel_averages[(iant, ichan)] = {
                            'sumFreqs': np.zeros(self.NFreqBins),
                            'sumdBm': np.zeros(self.NFreqBins),
                            'NEntries': 0
                        }
                    # Sum the frequencies and dBm values for each channel of each antenna
                    self.channel_averages[(iant, ichan)]['sumFreqs'] += np.array(freqs)
                    self.channel_averages[(iant, ichan)]['sumdBm'] += np.array(amps)
                    self.channel_averages[(iant, ichan)]['NEntries'] += 1

    def Physics(self, frame):
        self.SpectrumAverage(frame, "ArtifactsRemoved")

    def Finish(self):
        plt.figure(figsize=(20, 15))
        cmap = sns.color_palette("mako_r", as_cmap=True)
        color_map = cmap(np.linspace(0.2, 0.8, 3))
        for iant in range(3):
            for ichan in range(2):
                channel_data = self.channel_averages[(iant, ichan)]
                avg_freqs = channel_data['sumFreqs'] / channel_data['NEntries']
                avg_dBm = channel_data['sumdBm'] / channel_data['NEntries']
                # Create a subplot for the current antenna
                plt.subplot(3, 1, iant+1)
                plt.plot(avg_freqs/ I3Units.megahertz, avg_dBm, label=f"Antenna {iant+1}, Channel {ichan+1}", color=color_map[ichan])
                plt.ylim(120, 180)
                plt.xlim(0, np.max(avg_freqs/ I3Units.megahertz))
                x_ticks = np.arange(0, np.max(avg_freqs)/ I3Units.megahertz, 50)
                plt.grid(True)
                plt.xticks(x_ticks)
                plt.legend()
                plt.title(f"Spectral Average - Antenna {iant+1}")
                plt.ylabel("Spectral power [dBm/Hz]")

        plt.xlabel("Frequency [MHz]")
        plt.legend()
        plt.savefig("spect_aug_sept_2023.png", dpi=360)

init_time = time.time()
tray = I3Tray()

tray.AddModule("I3Reader", "reader",
         FilenameList = filename)

# Choosing soft trigger only
def select_soft(frame):
    trigger_info = frame['SurfaceFilters']
    return trigger_info["soft_flag"].condition_passed

# Add the module to the tray
tray.Add(select_soft, "select_soft",
         streams=[icetray.I3Frame.DAQ])

# Select data with trace length equal to 1024
def select_TraceLength(frame):
    TraceLength = frame['RadioTraceLength'].value

    if TraceLength == 1024:
        return True  # This will indicate that the frame should be saved to the selected stream
    else:
        return False

# Add the module to the tray
tray.Add(select_TraceLength, "select_TraceLength",
         streams=[icetray.I3Frame.DAQ])

tray.AddModule("I3NullSplitter","splitter",
               SubEventStreamName="RadioEvent"
               )

tray.AddModule(AnalyzeQframes, "Plotter")

tray.Execute()
print(f'-- Time elapsed: {(time.time()-init_time)/60:.2f} min.')