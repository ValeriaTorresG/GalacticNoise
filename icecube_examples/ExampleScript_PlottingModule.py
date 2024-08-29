#!/bin/env python3

#This script is a fairly comprehensive overview of all the tyical analysis modules in radcube
#The result is a plot that shows an individual waveform for an antenna after each step in the
#processing (AKA for each module, one plot is created)

#You can run this program as: 
#  python ExampleScript_PlottingModule.py /data/sim/IceCubeUpgrade/CosmicRay/Radio/coreas/data/continuous/star-pattern/proton/lgE_18.0/sin2_0.5/000026

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--antid', type=int, default=1, help='Choose which antenna ID to plot')
parser.add_argument('--stnid', type=int, default=-666, help='Choose which station ID to plot')
parser.add_argument('--output', type=str, default="./", help='Name of the output directory')
parser.add_argument('input', type=str, nargs='+', help='Input data files.')
args = parser.parse_args()


from icecube.icetray import I3Tray
from icecube import icetray, radcube, dataio
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units

bandLimits = [100*I3Units.megahertz, 200*I3Units.megahertz]
electronicName = "electronicResponseName"
antennaName = "antennaResponseName"

#If you are not working on Cobalt, you will need to tell the
#Electronics/antenna responses where to find the tabulated files
headDirOverride = "" 

tray = I3Tray()

################################
## Add services
################################

#Includes the tables of the electronics: LNA, radioboard, etc.
tray.AddService("I3ElectronicsResponseFactory", electronicName,
                DAQTemperature=radcube.constants.electronicsTemp,
                IncludeLNA=True,
                IncludeCables=True,
                CableTemperature=radcube.constants.cableTemp,
                AdditionalGain=0,
                IncludeDAQ=True,
                InstallServiceAs = electronicName,
                OverrideHeadDir=headDirOverride
               )

#Includes the tables of the antenna gain patterns
tray.AddService("I3AntennaResponseFactory", antennaName,
                AntennaType=radcube.constants.antennaType,
                InstallServiceAs=antennaName,
                OverrideHeadDir=headDirOverride
               )

#Gives you random numbers
tray.AddService("I3GSLRandomServiceFactory", "gslRandom",
                Seed=666,
                InstallServiceAs="gslRandom")


################################
## Add modules
################################

tray.AddModule("CoreasReader", "coreasReader",
               DirectoryList=args.input,
               MakeGCDFrames=True,
               MakeDAQFrames=True,
              )

#Add zeros to the front of the EFields so that the total length is
#5000 bins
tray.AddModule("ZeroPadder", "iPad",
               InputName="CoREASEFieldMap",
               OutputName="ZeroPaddedMap",
               ApplyInDAQ = True,
               AddToFront = True,
               AddToTimeSeries = True,
               FixedLength = 5000
              )

#Convolves the EFields with the antenna gain patterns. After this module
#all data will be I3AntennaDataMaps with voltages in them
tray.AddModule("ChannelInjector", "_ChannelInjector",
                InputName="ZeroPaddedMap",
                OutputName="RawVoltageMap",
                AntennaResponseName=antennaName
              )

#Adds a phase delay to the signals which "rotates" the bins
tray.AddModule("AddPhaseDelay", "AddPhaseDelay",
                InputName="RawVoltageMap",
                OutputName="PhaseDelayed",
                ApplyInDAQ=True,
                DelayTime=-270*I3Units.ns
              )

#Will change the sampling rate of a trace to a higher/lower one
tray.AddModule("TraceResampler", "Resampler",
               InputName="PhaseDelayed",
               OutputName="ResampledVoltageMap",
               ResampledBinning=radcube.constants.resampledBinning
              )

#Cuts the waveforms to be a smaller size, in this case 1000 bins long
tray.AddModule("WaveformChopper", "LilChoppy",
                InputName = "ResampledVoltageMap",
                OutputName = "ChoppedWaveformMap",
                MaxBin = 999,
                MinBin = 0
               )

#Adds uncorrelated Cane and/or thermal noise to the waveform.
#This also puts an I3AntennaDataMap into the frame with ONLY the injected noise
#called "GeneratedNoiseMap"
tray.AddModule("BringTheNoise", "NoiseGenerator",
               AntennaResponseName=antennaName,
               UseThermalNoise=True,
               ThermalNoiseTemp=radcube.constants.thermalNoiseTemp,
               UseCaneNoise=True,
               RandomServiceName="gslRandom",
               InputName="ChoppedWaveformMap",
               OutputName="NoisyWaveform"
              )

#Convolves the voltages with the electronics response
tray.AddModule("ElectronicResponseAdder", "AddElectronics",
               InputName="NoisyWaveform",
               OutputName="FoldedVoltageMap",
               ElectronicsResponse=electronicName
              )

#Changes the real-valued waveforms to ADC bits like TAXI would do
tray.AddModule("WaveformDigitizer", "waveformdigitizer",
               InputName="FoldedVoltageMap",
               OutputName="DigitizedVoltageMap",
               ElectronicsResponse=electronicName
              )

#Makes the P-Frame for this Q-Frame
tray.AddModule("I3NullSplitter","splitter",
               SubEventStreamName="RadioEvent"
               )

#Converts the ADC counts back into voltages and removes the baseline
tray.AddModule("PedestalRemover", "pedestalRemover",
               InputName="DigitizedVoltageMap",
               OutputName="PedestalRemoved",
               ElectronicsResponse=electronicName,
               ConvertToVoltage=True
              )

#Applies a filter to the data with the given limits and filter type
tray.AddModule("BandpassFilter", "BoxFilter",
               InputName="PedestalRemoved",
               OutputName="FilteredMap",
               FilterType=radcube.eBox,
               FilterLimits=bandLimits,
              )

#Deconvolves the voltages with the electronics response
tray.AddModule("ElectronicResponseRemover", "RemoveElectronics",
               InputName="FilteredMap",
               OutputName="DeconvolvedMap",
               ElectronicsResponse=electronicName
              )

tray.AddModule(radcube.modules.RadcubePlotter, "thePlotter",
    AntennaID  = args.antid,
    StationID  = args.stnid,
    OutputDir = args.output,
    ShowHilbert = True,
    DataToPlot = [["CoREASEFieldMap", 1, "Raw EFields", False, "raw"],\
                  ["ZeroPaddedMap", 1, "Zero Padded EFields", False, "raw"],\
                  ["RawVoltageMap", 1, "Raw Voltages in the Antenna", False, "dB"],\
                  ["PhaseDelayed", 1, "Delayed Signal", False, "dB"],\
                  ["ResampledVoltageMap", 1, "Resampled Signal", False, "dB"],\
                  ["ChoppedWaveformMap", 1, "Chopped Signal", False, "dB"],\
                  ["NoisyWaveform", 1, "Noisy Signal", False, "dB"],\
                  ["GeneratedNoiseMap", 1, "Injected Noise Only", False, "dB"],\
                  ["FoldedVoltageMap", 1, "Folded With Electronics", False, "dB"],\
                  ["DigitizedVoltageMap", 1, "Digitized TAXI Output", True, "linear"],\
                  ["PedestalRemoved", 1, "Undigitized TAXI Output", False, "dB"],\
                  ["FilteredMap", 1, "Frequency Filtered Signal", False, "dB"],\
                  ["DeconvolvedMap",1, "Electronics Deconvolved", False, "dB"],\
                  ]
    )


tray.Execute()