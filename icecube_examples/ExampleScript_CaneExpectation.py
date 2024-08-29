#!/bin/env python3

# This script makes a plot of the expected amplitude according to the Cane model
# plus thermal noise. Outputs a plot called CaneExpectation.pdf
# Run as: python ExampleScript_CaneExpectation.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from icecube import radcube, dataclasses, icetray
from icecube.icetray import I3Units
from icecube.icetray import I3Tray

icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)

# You can request waveforms with various binning and length
WaveformLengths = [1000]
Binning = [1 * I3Units.ns]

# Custom module to make waveforms with zero amplitude to bootstrap the
# BringTheNoise module
class MakeEmptyWaveforms(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)

    def Configure(self):
        self.NEntries = 0

    def Process(self):
        frame = icetray.I3Frame(icetray.I3Frame.DAQ)

        antMap = dataclasses.I3AntennaDataMap()

        geometry = dataclasses.I3Geometry()

        if self.NEntries < len(WaveformLengths):

            antkey = radcube.GetGenericAntennaKey(self.NEntries)

            antGeo = dataclasses.I3AntennaGeo()
            antGeo.cableLength = 50 * I3Units.meter
            geometry.antennageo[antkey] = antGeo

            antChMap = dataclasses.I3AntennaChannelMap()

            fft = dataclasses.FFTData()
            timeSeries = fft.GetTimeSeries()
            timeSeries.binning = 1.0 * I3Units.ns
            timeSeries.offset = 0.0

            for ibin in range(WaveformLengths[self.NEntries]):
                timeSeries.PushBack(0.0)

            antCh = dataclasses.I3AntennaChannel(fft)
            antChMap[0] = antCh
            antCh2 = dataclasses.I3AntennaChannel(fft)
            antChMap[1] = antCh2
            antMap[antkey] = antChMap

        frame["EmptyWaveform"] = antMap
        frame[radcube.GetDefaultGeometryName()] = geometry

        self.NEntries += 1

        self.PushFrame(frame)

        if self.NEntries == len(WaveformLengths):
            self.RequestSuspension()


class PlotExpectation(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.fig = plt.figure(figsize=(8, 5))
        self.gs = gridspec.GridSpec(len(WaveformLengths), 1, wspace=0.1, hspace=0.1)
        self.gsCount = 0

    def MakePlot(self, ax, name, frame):
        antDataMap = frame[name]
        antkey = antDataMap.keys()[0]

        chDataMap = antDataMap[antkey]
        chkey = chDataMap.keys()[0]

        fft = chDataMap[chkey].GetFFTData()

        spectrum = fft.GetFrequencySpectrum()

        freqs, amps = radcube.RadTraceToPythonList(spectrum)

        amps = [radcube.GetDbmHzFromFourierAmplitude(abs(thisAmp), spectrum.binning, 50 * I3Units.ohm) for thisAmp in amps]

        plotFreqs = []
        plotdBm = []

        for (freq, amp) in zip(freqs, amps):
            if amp > -200:
                plotFreqs.append(freq)
                plotdBm.append(amp)
                print("{0}, {1:0.3f},".format(freq / I3Units.megahertz, amp))

        ax.plot(np.array(plotFreqs) / I3Units.megahertz, plotdBm)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("dBm/Hz")
        ax.set_xlim(0, max(np.array(plotFreqs) / I3Units.megahertz) * 1.1)

    def DAQ(self, frame):
        if self.gsCount < len(WaveformLengths):
            ax = self.fig.add_subplot(self.gs[self.gsCount])
            self.gsCount += 1

            self.MakePlot(ax, "FoldedWaveforms", frame)

            print("Making file CaneExpectation.pdf")
            plt.savefig("CaneExpectation.pdf", bbox_inches="tight")
            plt.close()




tray = I3Tray()

antennaName = radcube.defaults.CreateDefaultAntennaResponse(tray)
electronicName = radcube.defaults.CreateDefaultElectronicsResponse(tray)

tray.AddService("I3GSLRandomServiceFactory", "gslRandom",
                Seed=666,
                InstallServiceAs="gslRandom")

tray.AddModule(MakeEmptyWaveforms, "Empty")

tray.AddModule("BringTheNoise", "NoiseGenerator",
               AntennaResponseName=antennaName,
               UseThermalNoise=True,
               UseCaneNoise=True,
               RandomServiceName="gslRandom",
               InputName="EmptyWaveform",
               OutputName="NoisyWaveform"
              )

tray.AddModule("ElectronicResponseAdder", "AddElectronics",
               InputName="NoisyWaveform",
               OutputName="FoldedWaveforms",
               ElectronicsResponse=electronicName
              )

tray.AddModule(PlotExpectation, "Plotter")

tray.Execute(1)