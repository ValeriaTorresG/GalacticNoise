#!/usr/bin/env python

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

from icecube.icetray import I3Tray
from icecube import icetray, radcube, dataio, dataclasses
from icecube.radcube import defaults
from icecube.icetray import I3Units
from icecube.icetray.i3logging import log_fatal, log_info
import numpy as np

import argparse

icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="+", help="Input data files.")
parser.add_argument("--output", dest="outfile", type=str, default="ArrivalTimes_Plot.pdf", help="Name of the plot")
args = parser.parse_args()

if len(args.input) > 1:
    log_fatal("This script is made for only one shower at a time. You have given: " + str(len(args.input)))



class PlottingModule(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.fig = 0
        self.gs = 0
        self.gsCount = 0

    def Configure(self):
        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(1, 1, wspace=0.1, hspace=0.1)

    def PlotArrivalTimes(self, ax, particle, antMap):
        for antkey in antMap.keys():
            channelMap = antMap[antkey]
            for chkey in channelMap.keys():
                channelData = channelMap[chkey]
                time = channelData.GetTimeAtMaximum()
                amplitude = channelData.GetMaxAmplitude()

    def Physics(self, frame):
        if "I3RadGeo" in frame:
            geo = I3RadGeo()
        else:
            log_fatal("Could not find the geometry in the frame" + str(frame))

        if "ReconPrep_EstimatedRadioShower" in frame:
            estimatedParticle = frame["ReconPrep_EstimatedRadioShower"]
        else:
            log_fatal("Could not find the estimated particle" + str(frame))

        if "CoREASPrimary" in frame:
            mcTruth = frame["CoREASPrimary"]
        else:
            log_fatal("Could not find the MC Primary" + str(frame))

        if "ReconPrep_Unfolded" in frame:
            antMap = frame["ReconPrep_Unfolded"]
        else:
            log_fatal("Could not fine AntennaDataMap in the frame" + str(frame))

        ax = self.fig.add_subplot(self.gs[self.gsCount])
        self.gsCount += 1

        radius = []
        arrivalTime = []

        for antkey in rawDataMap.keys():
            thisWaveform = rawDataMap[antkey]

            # Rotate all of the elements in the list
            # Do not actually need to do this, but just to show how...
            for ibin in range(thisWaveform.GetSize()):
                vec = thisWaveform.GetElem(ibin)
                vec = radcube.GetMagneticFromIC(vec, mcTruth.dir)  # Convert to Lorentz coordinates
                thisWaveform.SetElem(ibin, vec)  # Put the rotated back in

            fftDataContainer = radcube.FFTData3D()  # Make an fft container
            fftDataContainer.LoadTimeSeries(thisWaveform)  # Put the EFieldTimeSeries into it

            peakTime = radcube.GetHilbertPeakTime(fftDataContainer)  # Do not actually need to do this, but just to show how...

            arrivalTime.append(peakTime)

            # Get the antenna locations and save them
            antennaLoc = geomap[antkey].position
            antennaLoc.z = 0  # The IC origin is in the middle of in-ice, move it first to surface
            antennaLoc = radcube.GetMagneticFromIC(antennaLoc, mcTruth.dir)

            radius = np.sqrt(antennaLoc.x ** 2 + antennaLoc.y ** 2) / I3Units.m

        minRad = 10000000
        minTime = 1
        for iant in len(radius):
            if radius[iant] < minRad:
                minRad = radius[iant]
                minTime = arrivalTime[iant]

        arrivalTime = [val - minTime for val in arrivalTime]

        ax = self.fig.add_subplot(self.gs[self.gsCount])
        self.gsCount += 1

        # Plot the colors and sizes of markers based on the amplitude
        ax.scatter(radius, arrivalTime, s=amp, c=amp, label=("Arrival Time"))
        # ax.set_aspect('equal')
        ax.set_xlabel("Axial Radius [m]")
        ax.set_ylabel("Relative Arrival Time [ns]")

    def Finish(self):
        log_info("Making output file " + str(args.outfile))
        plt.savefig(args.outfile, bbox_inches="tight")
        plt.close()


def GetPerfectCore(frame):
    origParticle = frame["ReconPrep_EstimatedRadioShower"]
    newParticle = dataclasses.I3Particle(origParticle)
    mcTruth = frame["CoREASPrimary"]
    newParticle.pos = mcTruth.pos

    print("The MC Truth is", mcTruth)
    print("Putting in", newParticle)

    frame["PerfectCore"] = newParticle


tray = I3Tray()

ElectronicServiceName = defaults.CreateDefaultElectronicsResponse(tray)
RandomServiceName = defaults.CreateDefaultRandomService(tray)
AntennaServiceName = defaults.CreateDefaultAntennaResponse(tray)

tray.AddSegment(radcube.segments.RadioInjection, 'RadioInjection',
                 InputFiles = args.input,
                 ElectronicServiceName = ElectronicServiceName,
                 AntennaServiceName = AntennaServiceName,
                 RandomServiceName = RandomServiceName,
                 OutputName = "RadcubeInjectedVoltageMap"
                )

tray.AddModule("I3NullSplitter","splitter",
               SubEventStreamName="RadioEvent"
               )

tray.AddSegment(radcube.segments.RadioReconPrep, 'ReconPrep',
                InputName = "RadcubeInjectedVoltageMap",
                ElectronicServiceName = ElectronicServiceName,
                SNRThreshold = 20
               )

tray.AddModule(GetPerfectCore, "GetPerfectCore")
tray.AddModule(PlottingModule, "ThePlotter")

tray.Execute()