
#!/bin/env python3

#Runs a standard set of modules to produce a simulation. One of these loads up and includes measured noise waveforms from the Pole
#Makes a plot of one antenna's waveforms after each module

#You can run this program as: 
#  python ExampleScript_AddingMeasuredNoise.py /data/sim/IceCubeUpgrade/CosmicRay/Radio/coreas/data/continuous/star-pattern/proton/lgE_18.0/sin2_0.5/000026

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

#This is the TAXI file that will be used to add measured noise to the data
TaxiFile = "/data/exp/IceCube/2021/unbiased/surface/V6/sae_data/SAE_data_135038_1_IT.i3.gz"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--antid', type=int, default=1, help='Choose which antenna ID to plot')
parser.add_argument('--stnid', type=int, default=-666, help='Choose which station ID to plot')
parser.add_argument('--output', type=str, default="./", help='Directory for the output file')
parser.add_argument('input', type=str, nargs='+', help='Input data files.')
args = parser.parse_args()


from icecube.icetray import I3Tray
from icecube import icetray, radcube, dataio, dataclasses
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units
from icecube.icetray.i3logging import log_info, log_warn, log_debug
icetray.I3Logger.global_logger.set_level(icetray.I3LogLevel.LOG_INFO)
import numpy as np

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

#Convolves the voltages with the electronics response
tray.AddModule("ElectronicResponseAdder", "AddElectronics",
               InputName="ChoppedWaveformMap",
               OutputName="FoldedVoltageMap",
               ElectronicsResponse=electronicName
              )

#Changes the real-valued waveforms to ADC bits like TAXI would do
tray.AddModule("WaveformDigitizer", "waveformdigitizer",
               InputName="FoldedVoltageMap",
               OutputName="DigitizedVoltageMap",
               ElectronicsResponse=electronicName
              )

#Add measured noise waveforms to the simulated ones
tray.AddModule(radcube.modules.MeasuredNoiseAdder, "addMeasuredNoise",
               InputName="DigitizedVoltageMap",
               OutputName="NoisyWaveform",
               ConvertToVoltage=False, #Leave in ADC units
               NTimeBins=1000, #Also cut down to 1000 bins
               NTraces=200,  #Only bother to load 200 waveforms
               TaxiFile=TaxiFile,
               InsertNoiseOnly=True, #Also add the pure noise to the frame
               Overuse=True, #Allow to use each waveform more than once, if needed
               MedianOverCascades=False, #Average over non-cascaded copies
               RemoveBinSpikes=True, # Remove TAXI artifact
               RemoveNegativeBins=True, # Remove TAXI artifact
               MatchExactNoise=False, #Do not match measured noise to the simulated antennas
              )

#Makes the P-Frame for this Q-Frame
tray.AddModule("I3NullSplitter","splitter",
               SubEventStreamName="RadioEvent"
               )

#Converts the ADC counts back into voltages and removes the baseline
tray.AddModule("PedestalRemover", "pedestalRemover",
               InputName="NoisyWaveform",
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

tray.AddModule("I3Writer", "Writer",
               filename = "output.i3.gz")

tray.AddModule(radcube.modules.RadcubePlotter, "thePlotter",
    AntennaID  = args.antid,
    StationID  = args.stnid,
    OutputDir = args.output,
    ZoomToPulse = 0,
    DataToPlot = [["CoREASEFieldMap", 1, "Raw EFields", False, "raw"],\
                  ["ZeroPaddedMap", 1, "Zero Padded EFields", False, "raw"],\
                  ["RawVoltageMap", 1, "Raw Voltages in the Antenna", False, "dB"],\
                  ["PhaseDelayed", 1, "Delayed Signal", False, "dB"],\
                  ["ResampledVoltageMap", 1, "Resampled Signal", False, "dB"],\
                  ["ChoppedWaveformMap", 1, "Chopped Signal", False, "dB"],\
                  ["FoldedVoltageMap", 1, "Folded With Electronics", False, "dB"],\
                  ["DigitizedVoltageMap", 1, "Digitized TAXI Output", True, "linear"],\
                  ["NoisyWaveform", 1, "Noisy Signal", True, "linear"],\
                  ["TAXINoiseMap", 1, "Injected Noise Only", True, "linear"],\
                  ["PedestalRemoved", 1, "Undigitized TAXI Output", False, "dB"],\
                  ["FilteredMap", 1, "Frequency Filtered Signal", False, "dB"],\
                  ["DeconvolvedMap",1, "Electronics Deconvolved", False, "dB"],\
                  ]
    )


tray.Execute()
