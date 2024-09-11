#!/usr/bin/env python3

from icecube.icetray import I3Units
from icecube.taxi_reader import taxi_tools
from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, taxi_reader, radcube
from icecube.icetray.i3logging import log_info
import matplotlib.gridspec as gridspec

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse


'''#Input i3 file with the data
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="+", default=[], help="List of i3 files")
args = parser.parse_args()

filename = args.input'''
#Input i3 file with the data

#filename=["/data/user/storres/GalacticNoiseAnalysisIceCube/RadioDataIcecube/i3_files/eventData_1609493854_2021-01-01_09-37-34_134851.i3.gz"]

#filename = ['/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704413407_2024-01-05_00-10-07_4095_138820.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704413407_2024-01-05_00-10-07_4095_138821.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704413407_2024-01-05_00-10-07_4095_138822.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704499807_2024-01-06_00-10-07_4095_138823.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704499807_2024-01-06_00-10-07_4095_138824.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704499807_2024-01-06_00-10-07_4095_138826.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704499807_2024-01-06_00-10-07_4095_138827.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704499807_2024-01-06_00-10-07_4095_138828.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704586206_2024-01-07_00-10-06_4095_138829.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704586206_2024-01-07_00-10-06_4095_138830.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704586206_2024-01-07_00-10-06_4095_138831.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704672607_2024-01-08_00-10-07_4095_138832.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704672607_2024-01-08_00-10-07_4095_138833.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704672607_2024-01-08_00-10-07_4095_138834.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704759007_2024-01-09_00-10-07_4095_138835.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704759007_2024-01-09_00-10-07_4095_138836.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704759007_2024-01-09_00-10-07_4095_138837.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704845406_2024-01-10_00-10-06_4095_138838.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704845406_2024-01-10_00-10-06_4095_138843.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704845406_2024-01-10_00-10-06_4095_138847.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704845406_2024-01-10_00-10-06_4095_138848.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704845406_2024-01-10_00-10-06_4095_138849.i3.gz','/mnt/lfs6/exp/IceCube/2024/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1704845406_2024-01-10_00-10-06_4095_138850.i3.gz']
filename = ['/mnt/ceph1-npx/user/valeriatorres/galactic_noise/SouthPole/i3_files/processed_2024-01-09.i3.gz']
WaveformLengths = [1024]

# This cla'ss take the spectrum average over all the Q frames
class AnalyzeQframes(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.NEntries = 0
        self.NFreqBins = int(WaveformLengths[0]/ 2 + 1)
        self.plotdBm = np.zeros(self.NFreqBins)
        self.plotFreqs = np.zeros(self.NFreqBins)
        self.channel_averages = {}  # Dictionary to store the average spectrum for each channel

    def SpectrumAverage(self, frame, name):
        if frame.Has("MedFilteredMap"):
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
        self.SpectrumAverage(frame, "MedFilteredMap")

    def Finish(self):
        cmap = sns.color_palette("mako_r", as_cmap=True)
        color_map = cmap(np.linspace(0.2, 0.8, 3))
        plt.figure(figsize=(20, 15))
        for iant in range(3):
            for ichan in range(2):
                channel_data = self.channel_averages[(iant, ichan)]
                avg_freqs = channel_data['sumFreqs'] / channel_data['NEntries']
                avg_dBm = channel_data['sumdBm'] / channel_data['NEntries']
                # Create a subplot for the current antenna
                plt.subplot(3, 1, iant+1)
                if ichan == 0:
                    plt.plot(avg_freqs/ I3Units.megahertz, avg_dBm, label=f"Antenna {iant+1}, Channel {ichan+1}", color=color_map[ichan])
                else:
                    plt.plot(avg_freqs/ I3Units.megahertz, avg_dBm, label=f"Antenna {iant+1}, Channel {ichan+1}", color=color_map[ichan])

                plt.xlim(0, np.max(avg_freqs/ I3Units.megahertz))
                interval = 10
                x_ticks = np.arange(0, np.max(avg_freqs)/ I3Units.megahertz, interval)
                plt.xticks(x_ticks)
                plt.title(f"Spectral Average - Antenna {iant+1}")
                plt.ylabel("Spectral power [dBm/Hz]")
                plt.legend()

        plt.xlabel("Frequency [MHz]")
        plt.savefig("spc_median.png")


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

# # Removing TAXI artifacts
# tray.Add(
#     radcube.modules.RemoveTAXIArtifacts, "ArtifactRemover",
#     InputName="RadioTAXIWaveform",
#     OutputName="ArtifactsRemoved",
#     medianOverCascades=True,
#     BaselineValue=0,
#     RemoveBinSpikes=True,
#     BinSpikeDeviance=int(2**12),
#     RemoveNegativeBins=True
#     )

tray.AddModule("I3NullSplitter","splitter",
               SubEventStreamName="RadioEvent"
               )
# tray.AddModule("MedianFrequencyFilter", "MedianFilter_",
#             InputName="ArtifactsRemoved",
#             FilterWindowWidth=20,
#             OutputName="MedFilteredMap")

tray.AddModule(AnalyzeQframes, "Plotter")

tray.Execute()