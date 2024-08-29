#!/bin/env python3

# This script shows you how you can read in a CoREAS file to I3 format
# and how to do some basic data augmentation to that data
# This script will plot the locations of the stations and the signal
# strength of the electric field in the VxB coordinate system
# Run as: python ExampleScript_GetEFieldInVxB.py /data/sim/IceCubeUpgrade/CosmicRay/Radio/coreas/data/continuous/star-pattern/proton/lgE_18.0/sin2_0.5/000026

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, nargs="+", help="Input data files.")
parser.add_argument("--output", dest="outfile", type=str, default="GetEFieldInVxB_Plot.pdf", help="Name of the plot")
args = parser.parse_args()

if len(args.input) > 1:
    log_fatal("This script is made for only one shower at a time. You have given: " + str(len(args.input)))

from icecube.icetray import I3Tray
from icecube import icetray, radcube, dataio, dataclasses
from icecube.icetray import I3Units
from icecube.dataclasses import I3Constants
from icecube.icetray.i3logging import log_fatal, log_warn, log_info
import numpy as np


class PlottingModule(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.gsCount = 0

    def Configure(self):
        self.fig = plt.figure(figsize=(11, 5))
        self.gs = gridspec.GridSpec(1, 2, wspace=0.3, hspace=0.3)

    def DAQ(self, frame):
        if frame.Has("CoREASEFieldMap"):
            rawDataMap = frame["CoREASEFieldMap"]  # Get the simulated traces
        else:
            log_fatal("No fields found in the frame")

        if frame.Has("CoREASPrimary"):
            primary = frame["CoREASPrimary"]  # Get the primary particle direction
        else:
            log_fatal("No primary found in the frame")
        showerDir = primary.dir

        if frame.Has("I3AntennaGeometry"):
            geomap = frame["I3AntennaGeometry"].antennageo
        else:
            log_fatal("No geometry found in the frame")

        xPos = []
        yPos = []
        amp = []
        fluence = []

        for antkey in rawDataMap.keys():
            thisWaveform = rawDataMap[antkey]

            # Rotate all of the elements in the list
            # Do not actually need to do this, but just to show how...
            for ibin in range(thisWaveform.GetSize()):
                vec = thisWaveform[ibin]
                vec = radcube.GetMagneticFromIC(vec, showerDir)  # Convert to Lorentz coordinates
                thisWaveform[ibin] = vec  # Put the rotated back in

            fftDataContainer = dataclasses.FFTData3D()  # Make an fft container
            fftDataContainer.LoadTimeSeries(thisWaveform)  # Put the EFieldTimeSeries into it

            peak = dataclasses.fft.GetHilbertPeak(fftDataContainer)
            flu = radcube.GetEnergy(fftDataContainer.GetTimeSeries(), I3Constants.z_vacuum)
            peakTime = dataclasses.fft.GetHilbertPeakTime(fftDataContainer)  # Do not actually need to do this, but just to show how...

            if peak <= 0:  # Check to make sure that we don't take a log of something we should not
                log_warn("Found a strange peak of" + str(peak))
                continue
            amp.append(peak)

            # Get the antenna locations and save them
            antennaLoc = geomap[antkey].position
            antennaLoc.z -= primary.pos.z  # The IC origin is in the middle of in-ice, move it first to surface
            antennaLoc = radcube.GetMagneticFromIC(antennaLoc, showerDir)
            xPos.append(antennaLoc.x / I3Units.m)
            yPos.append(antennaLoc.y / I3Units.m)

        # Do a bit of scaling so that we have markers that are visible
        amps = np.array(amp)
        peakScaled = np.log10(amps / (1e-6 * I3Units.volt / I3Units.m))
        amps = np.log10(amps / (1e-6 * I3Units.volt / I3Units.m))
        minpeak = min(peakScaled)
        maxpeak = max(peakScaled)
        for i, val in enumerate(peakScaled):
            val = 20 * (val - minpeak) / (maxpeak - minpeak)
            peakScaled[i] = val

        ax = self.fig.add_subplot(self.gs[self.gsCount])
        self.gsCount += 1

        # Plot the colors and sizes of markers based on the amplitude
        scat = ax.scatter(xPos, yPos, s=peakScaled, c=amp, label=("Raw E(t) [uV/m] IC-E"))
        ax.set_aspect("equal")
        ax.set_xlabel("VxB [m]")
        ax.set_ylabel("VxVxB [m]")
        ax.set_title("Lg(E/eV): {0:0.1f}, Zen: {1:0.1f} deg".format(np.log10(primary.energy / I3Units.eV), showerDir.zenith / I3Units.degree))

        cbar = plt.colorbar(scat, shrink=0.8)
        cbar.set_label(r"Peak Amplitude lg(A / (uV/m))")

        showerCoreVxB = radcube.GetMagneticFromIC(dataclasses.I3Position(primary.pos.x, primary.pos.y, 0), showerDir)

        insetX = []
        insetY = []
        insetAmp = []

        imax = np.argmax(peakScaled)
        maxRad = np.sqrt((xPos[imax] - showerCoreVxB.x) ** 2 + (yPos[imax] - showerCoreVxB.y) ** 2)
        print("Radius of maximum amplitude is", maxRad)
        print("Zenith angle of:", primary.dir.zenith / I3Units.degree)

        maxRad *= 1.2
        if maxRad < 400:
            maxRad = 400

        for i in range(len(peakScaled)):
            if abs(xPos[i] - showerCoreVxB.x) < maxRad and abs(yPos[i] - showerCoreVxB.y) < maxRad:
                insetX.append(xPos[i])
                insetY.append(yPos[i])
                insetAmp.append(peakScaled[i])

        ax = self.fig.add_subplot(self.gs[self.gsCount])
        self.gsCount += 1

        if showerCoreVxB.x != 0 and showerCoreVxB.y != 0:
            ax.axvline(showerCoreVxB.x)
            ax.axhline(showerCoreVxB.y)

        scat2 = ax.scatter(insetX, insetY, s=insetAmp, c=insetAmp, label=("Raw E(t) [uV/m] IC-E"))
        ax.set_aspect("equal")
        ax.set_xlabel("VxB [m]")
        ax.set_ylabel("VxVxB [m]")
        ax.set_title("Lg(E/eV): {0:0.1f}, Zen: {1:0.1f} deg".format(np.log10(primary.energy / I3Units.eV), showerDir.zenith / I3Units.degree))

        cbar2 = plt.colorbar(scat2, shrink=0.8)
        cbar2.set_label(r"Peak Amplitude lg(A / (uV/m))")

    def Finish(self):
        log_info("Making output file " + str(args.outfile))
        plt.savefig(args.outfile, bbox_inches="tight")
        plt.close()


tray = I3Tray()

tray.AddModule("CoreasReader", "CoreasReader",
               DirectoryList=args.input,      #List of CoREAS output directories (one per shower)
               MakeGCDFrames=True,            #Make a GCD frames from simulations
               MakeDAQFrames=True,            #Make the Q frames from simulations
              )

tray.AddModule(PlottingModule, "ThePlotter")

tray.Execute()