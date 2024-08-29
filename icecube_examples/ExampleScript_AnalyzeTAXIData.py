#!/bin/env python3

# This script is used to show how to do some basic calculations on the TAXI waveforms
# This assumes you already have an I3 file created from measured data:
# python ExampleScript_AnalyzeTAXIData.py /data/exp/IceCube/2021/unbiased/surface/V6/radio_temp/i3_files/eventData_1618917599_2021-04-20_11-19-59_135209.i3.gz

# An I3 file can alternatively be created from the binary files the TAXI outputs, using the taxi-reader project.

import numpy as np

from icecube.icetray import I3Units
from icecube.taxi_reader import taxi_tools

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, taxi_reader, radcube
from icecube.icetray.i3logging import log_info


class AnalyzeTAXIData(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)

    def Configure(self):

        self.NAntennas = 3
        self.NArms = 2
        self.NTimeBins = 1000
        self.NFreqBins = int(self.NTimeBins / 2 + 1)

        self.AverageSpectrum = np.zeros((self.NAntennas, self.NArms, self.NFreqBins))
        self.AverageTimeSeries = np.zeros((self.NAntennas, self.NArms, self.NTimeBins))

        self.NEntries = 0
        self.df = -1

        self.verbose = True

    def DAQ(self, frame):
        log_info("This function runs once per frame")


        antennaDataMap = frame["TAXICleaned"]  # This is a container that holds all the antenna info

        for iant, antkey in enumerate(antennaDataMap.keys()):

            if self.verbose:
                log_info("Working on antenna for antkey " + str(antkey))

            channelMap = antennaDataMap[antkey]  # This container holds all information for one antenna

            for ichannel, chkey in enumerate(channelMap.keys()):  # Each antenna arm is one "channel"

                if self.verbose:
                    log_info("Working on channel " + str(chkey))

                fft = channelMap[ichannel].GetFFTData()  # This container holds time series and spectrum

                #################
                ##Analyze the time series
                #################
                timeSeries = fft.GetTimeSeries()  # Get the time series for this antenna
                timeSeries *= taxi_tools.get_volts_per_ADC_bin()  # Convert it from ADC bins to volts!

                # Get just a chunk of the time series, skip the first time bin
                truncatedTimeSeries = timeSeries.GetSubset(1, self.NTimeBins)

                # Let's first remove the offset
                truncatedTimeSeries -= radcube.GetMean(truncatedTimeSeries)

                if self.verbose:
                    meanValueInmV = radcube.GetMean(truncatedTimeSeries) / I3Units.mV
                    rmsValueInmV = radcube.GetRMS(truncatedTimeSeries) / I3Units.mV
                    log_info(
                        "Analyzing time series with length {0} bins, mean value {1:0.2f} [mV], and RMS of {2:0.02f} [mV]".format(
                            len(truncatedTimeSeries), meanValueInmV, rmsValueInmV
                        )
                    )

                # Convert to python list of times and amplitudes
                binTimes, pythonWaveform = radcube.RadTraceToPythonList(truncatedTimeSeries)

                self.AverageTimeSeries[iant][ichannel] += np.array(pythonWaveform)

                #################
                ##Analyze the spectrum
                #################
                spectrum = fft.GetFrequencySpectrum()
                self.df = spectrum.binning  # We will need this for later

                truncatedFreqSpec = spectrum.GetSubset(0, self.NFreqBins - 1)
                freqs, pythonWaveform = radcube.RadTraceToPythonList(truncatedFreqSpec)

                # we don't care about the phases, so cast as magnitude
                pythonWaveform = np.abs(pythonWaveform)

                self.AverageSpectrum[iant][ichannel] += pythonWaveform

        self.NEntries += 1

    def Finish(self):
        log_info("We have grabbed information from all the events!")

        # Normalize the data
        self.AverageSpectrum /= self.NEntries
        self.AverageTimeSeries /= self.NEntries

        for iant in range(len(self.AverageSpectrum)):

            for ichannel in range(len(self.AverageSpectrum[iant])):

                print("For antenna {0} and channel {1}".format(iant + 1, ichannel))

                stdInmV = np.std(self.AverageTimeSeries[iant][ichannel]) / I3Units.mV
                print("....The time series standard deviation is {0:0.2f} [mV]".format(stdInmV))

                fourierAmp = self.AverageSpectrum[iant][ichannel][100]
                dBmHz = radcube.GetDbmHzFromFourierAmplitude(fourierAmp, self.df, 50 * I3Units.ohm)
                print("....The power at 200MHz is {0:0.2f} dBm/Hz".format(dBmHz))



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, default=[], help='Input data file. (I3 format)')
args = parser.parse_args()

tray = I3Tray()

tray.Add('I3Reader', "MyReader",
        FilenameList = [args.input]
      )

# Remove the readout effects/artifacts from TAXI data
tray.AddModule(radcube.modules.RemoveTAXIArtifacts, "ArtifactRemover",
              InputName=taxi_reader.taxi_tools.taxi_antenna_frame_name(), 
              OutputName="TAXICleaned",
              )

tray.AddModule(AnalyzeTAXIData, "ThisNameDoesNotMatter")

tray.Execute(10)