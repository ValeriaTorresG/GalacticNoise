#!/bin/env python3

# This script shows a small example on how to make and augment waveforms
# Specifically it plots three functions and the convolution of all
# pairs. Outputs the plot Convolution.pdf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from icecube import radcube, dataclasses
from icecube.radcube import defaults


###########################################
######## Basic Functions to Plot ##########
###########################################

NCounts = 3000  # Length of the waveforms in bins


def Func1():  # Square pulse
    out = dataclasses.AntennaTimeSeries()
    for ibin in range(NCounts):
        if ibin < int(NCounts / 4) or ibin > int(NCounts * 3 / 4):
            out.PushBack(0.0)
        else:
            out.PushBack(1.0)
    out.binning = 1
    out.offset = 0
    return out


def Func2():  # Saw-tooth pulse
    out = dataclasses.AntennaTimeSeries()
    for ibin in range(NCounts):
        if ibin < int(NCounts / 4) or ibin > int(NCounts * 3 / 4):
            out.PushBack(0.0)
        else:
            out.PushBack(1 - 1 / (NCounts / 2) * (ibin - NCounts / 4))
    out.binning = 1
    out.offset = 0
    return out


def Func3():  # Damped oscillation
    out = dataclasses.AntennaTimeSeries()
    for ibin in range(NCounts):
        out.PushBack(np.cos(ibin * 8 * 2 * np.pi / NCounts) * np.exp(-ibin / (NCounts / 4)))
    out.binning = 1
    out.offset = 0
    return out


###########################################
############# Plotting Tools ##############
###########################################


def PlotFunction(ax, timeSeries, **kargs):
    times, amps = radcube.RadTraceToPythonList(timeSeries)
    ax.plot(amps, **kargs)
    ax.legend(loc="upper right", prop={"size": 8})


def PlotConvolution(ax, timeSeries1, timeSeries2, **kargs):
    convolution = radcube.TimeSeriesConvolution(timeSeries1, timeSeries2)
    times, amps = radcube.RadTraceToPythonList(convolution)
    ax.plot(amps, **kargs)
    ax.legend(loc="upper right", prop={"size": 8})


allFuncs = [Func1, Func2, Func3]
colors = ["k", "r", "b", "g"]
fig = plt.figure(figsize=(6 * len(allFuncs), 5 * (1 + len(allFuncs))))
gs = gridspec.GridSpec(1 + len(allFuncs), len(allFuncs), wspace=0.2, hspace=0.2)

igs = 0
for i, iFunc in enumerate(allFuncs):
    ax = fig.add_subplot(gs[igs])
    igs += 1
    PlotFunction(ax, iFunc(), label="Function {0}".format(i + 1), color="k", alpha=0.5)

for i, iFunc in enumerate(allFuncs):
    for j, jFunc in enumerate(allFuncs):
        ax = fig.add_subplot(gs[igs])
        igs += 1
        color = colors[int(abs(i - j))]
        PlotConvolution(ax, iFunc(), jFunc(), label="Convolution F{0}*F{1}".format(i + 1, j + 1), color=color)

plt.savefig("Convolution.pdf", bbox_inches="tight")