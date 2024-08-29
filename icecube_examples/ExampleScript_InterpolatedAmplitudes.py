#!/bin/env python3

# This script uses a star-shaped simulation and interpolates the values for some specified range.
# Outputs a plot of the interpolated footprint with size specified by the inputs.
# Run as: python ExampleScript_InterpolatedAmplitudes.py /data/sim/IceCubeUpgrade/CosmicRay/Radio/coreas/data/continuous/star-pattern/proton/lgE_16.5/sin2_0.7/000000

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os

from icecube import radcube, dataclasses, icetray

icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
from icecube.icetray import I3Units

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Input data files.")
parser.add_argument("--output", type=str, default="InterpolatedLDF.pdf", help="The location of the snr cuts file")
parser.add_argument("--maxrad", type=float, default=300, help="The final plot will show the footprint on a this x this square")
parser.add_argument("--flow", type=float, default=100, help="Low frequency cut, given in MHz")
parser.add_argument("--fhigh", type=float, default=350, help="High frequency cut, given in MHz")
args = parser.parse_args()

args.flow *= I3Units.megahertz
args.fhigh *= I3Units.megahertz


assert os.path.isdir(args.input)


def ConvertToAntGeoMap(xVals, yVals):
    antennageomap = dataclasses.I3AntennaGeoMap()

    for x, y in zip(xVals, yVals):
        antgeo = dataclasses.I3AntennaGeo()
        antgeo.position = dataclasses.I3Position(x, y, 0.0)

        antennageomap[dataclasses.AntennaKey(len(antennageomap) + 1, len(antennageomap) + 1)] = antgeo

    return antennageomap


def ConvertFromAntGeoMap(keys, antennageomap):
    xVals = []
    yVals = []
    for key in keys:
        xVals.append(antennageomap[key].position.x)
        yVals.append(antennageomap[key].position.y)

    return xVals, yVals


# Read in all the data from the CoREAS simulation directory
primary = radcube.GetParticleFromCoREAS(args.input)
showerInfo = radcube.GetCorsikaShowerInfoFromCoREAS(args.input)
eFieldMap = radcube.GetEFiledMapFromCoREAS(args.input)
i3geo = dataclasses.I3Geometry()
i3cal = dataclasses.I3VEMCalibration()
i3det = dataclasses.I3DetectorStatus()
radcube.GetGCDFromCoREAS(args.input, i3geo, i3cal, i3det)
starPattern = radcube.StarInterpolator(eFieldMap, primary, i3geo.antennageo)


# We have to make the histogram by hand, get the bin edges
xEdge = np.linspace(-args.maxrad, args.maxrad, int(2 * args.maxrad / 10))
yEdge = np.linspace(-args.maxrad, args.maxrad, int(2 * args.maxrad / 10))

# Get the bin centers as well
xVals, yVals = np.meshgrid(
    [0.5 * (xEdge[i + 1] + xEdge[i]) for i in range(len(xEdge) - 1)],
    [0.5 * (yEdge[i + 1] + yEdge[i]) for i in range(len(yEdge) - 1)]
)
xVals = xVals.flatten()
yVals = yVals.flatten()

# Use the built in function to interpolate the response function
antennageomap = ConvertToAntGeoMap(xVals, yVals)
print("Interpolating. This may take a few min....")
eFieldMap = starPattern.GetEFieldMapAtTarget(antennageomap, False)
xVals, yVals = ConvertFromAntGeoMap(eFieldMap.keys(), antennageomap)

# Filter the waveforms to the specified band and calculate the energy
energies = []
print("Calculating energy at locations...")
for key, eField in eFieldMap:
    opts = radcube.FilterOptions()
    opts.SetBand(args.flow, args.fhigh)
    opts.type = radcube.eBox

    fft = dataclasses.FFTData3D(eField)
    filteredSpectrum = radcube.GetFilteredSpectrum(fft.GetFrequencySpectrum(), opts)

    energies.append(radcube.GetEnergy(filteredSpectrum, dataclasses.I3Constants.z_vacuum))

logE = np.log10(np.array(energies) / (I3Units.eV / I3Units.m2))

# Make the plot, etc.
NRows = 2
NCols = 2
gs = gridspec.GridSpec(NRows, NCols, wspace=0.3, hspace=0.3)
fig = plt.figure(figsize=(NCols * 6, NRows * 5))

ax = fig.add_subplot(gs[0])
hist = ax.hist2d(xVals, yVals, weights=logE, bins=(xEdge, yEdge))
ax.set_aspect("equal")
ax.set_title("Zenith: {0:0.1f}, {1:0.0f} - {2:0.0f} MHz".format(primary.dir.zenith * 180 / np.pi, args.flow / I3Units.megahertz, args.fhigh / I3Units.megahertz))
ax.set_xlabel("VxB / m")
ax.set_ylabel("VxBxB / m")

cbar = plt.colorbar(hist[3])
cbar.ax.set_ylabel(r"lg[E / (eV/m$^2$)]")

filename = args.output
fig.savefig(filename, bbox_inches="tight")
print("Saved", filename)