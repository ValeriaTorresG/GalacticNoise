#!/bin/env python3

# This program makes a plot of all the electronics response and shows how the
# I3ElectronicsResponse class works in case you ever wanted to use is manually.
# This will produce a plot called ElectronicsResponse.pdf in your currect
# directory with the gains for each component.
# Run as: python ExampleScript_ElectronicsResponse.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os

from icecube import radcube, dataclasses
from icecube.radcube import defaults
from icecube.icetray import I3Units

# The response function will give a zero for
# frequencies out of its tabulated range to this
# function removes these so that you can take a log
def ConvertToDb(freqs, amps):
    thisFreqs = []
    thisAmps = []
    for iamp, amp in enumerate(amps):
        if amp > 0:
            thisFreqs.append(freqs[iamp])
            thisAmps.append(20 * np.log10(amp))

    return np.array(thisFreqs), np.array(thisAmps)


# Here we go!

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

lnaOffset = -40
cableLength = 50 * I3Units.meter

# Define an I3AntennaCal object with specific electronics components
# which can be passed to the GetTotalResponse function of I3ElectronicsResponse
antCal = dataclasses.I3AntennaCal()
antCal.antennaType = dataclasses.I3AntennaCal.AntennaType.SKALA2
antCal.cableType = dataclasses.I3AntennaCal.CableType.LMR400
antCal.cableLength = 50 * I3Units.meter
antCal.daqType = dataclasses.I3AntennaCal.DaqType.Board1_Taxi3_0

df = 1 * I3Units.megahertz
freqs = np.array([i * df for i in range(501)])

# Initiate the class. By default every component is
# "turned off" so you have to tell the response class
# to include that component
response = radcube.I3ElectronicsResponse()

############## LNA Response ##############
response.IncludeLNA(True)
lnaAmps = []
for freq in freqs:
    lnaAmps.append(abs(response.GetLNAResponse(freq, antCal)))
plotFreqs, plotAmps = ConvertToDb(freqs, lnaAmps)
ax.plot(plotFreqs / I3Units.megahertz, plotAmps + lnaOffset, alpha=0.7, label="LNA " + str(lnaOffset) + "dB")
response.IncludeLNA(False)

############## Cable Response ##############
response.IncludeCables(True)
response.SetCableTemperature(radcube.constants.cableTemp)
cableAmps = []
for freq in freqs:
    cableAmps.append(abs(response.GetCableResponse(freq, antCal)))
plotFreqs, plotAmps = ConvertToDb(freqs, cableAmps)
ax.plot(plotFreqs / I3Units.megahertz, plotAmps, alpha=0.7, label="Cables")
response.IncludeCables(False)

# ########  Radio Board + TAXI Response #########
response.IncludeDAQ(True)
daqAmps = []
for freq in freqs:
    daqAmps.append(abs(response.GetDAQResponse(freq, antCal)))
plotFreqs, plotAmps = ConvertToDb(freqs, daqAmps)
ax.plot(plotFreqs / I3Units.megahertz, plotAmps, alpha=0.7, label="RadioBoard+TAXI")
response.IncludeDAQ(False)

########  Custom Component Response #########
comp = radcube.I3ComponentResponse(os.environ["I3_TESTDATA"] + "/radcube/electronic-response/Dummy1DResponse.txt", 1)
comp.SetName("Dummy Component")
response.AddCustomResponse(comp)
customAmps = []
for freq in freqs:
    customAmps.append(abs(response.GetTotalResponse(freq, antCal)))
plotFreqs, plotAmps = ConvertToDb(freqs, customAmps)
ax.plot(plotFreqs / I3Units.megahertz, plotAmps, alpha=0.7, label=comp.GetName())
response.ClearCustomResponses()

############## The Combined Response ##############
response.IncludeLNA(True)
response.IncludeCables(True)
response.IncludeDAQ(True)

totalAmps = []
for freq in freqs:
    totalAmps.append(abs(response.GetTotalResponse(freq, antCal)))
plotFreqs, plotAmps = ConvertToDb(freqs, totalAmps)
ax.plot(plotFreqs / I3Units.megahertz, plotAmps + lnaOffset, label="Total Response " + str(lnaOffset) + "dB", color="k")


ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Gain [dB]")
ax.legend(loc="best", prop={"size": 8})

# Some of the gains are go very low in dB...
ax.set_ylim(-15, 7)

print("Making file ElectronicsResponse.pdf")
plt.savefig("ElectronicsResponse.pdf", bbox_inches="tight")
plt.close()