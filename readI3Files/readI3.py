import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import glob
import time
import numpy as np
import astropy.time
import astropy.coordinates
from datetime import datetime
import logging

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, radcube
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units
from icecube import astro

# from utils import spectrum
# python readI3.py -stD "10" -stM "02" -enD "10" -enM "03" -y "2024"

parser = argparse.ArgumentParser(description='Read I3 files')
parser.add_argument('--startDay','-stD', type=str, help='Day',required=True)
parser.add_argument('--startMonth','-stM', type=str, help='Month',required=True)
parser.add_argument('--endDay','-enD', type=str, help='Day',required=True)
parser.add_argument('--endMonth','-enM', type=str, help='Month',required=True)
parser.add_argument('--year','-y', type=str, help='Year',required=True)
#=== Pulse Selection ===
parser.add_argument('--traceBins','-tb', type=int, help='Number of Bins in Time-trace',default=1024)
parser.add_argument('--triggerMode','-tM', help='Trigger Mode',default='noncascaded')
#=== Spectrum Extraction === # This hasn't been used yet. See analyzeFrame.py in utils. For future use.
parser.add_argument('--getTimeSeries','-gTS', help='Function to get Time Series',default=False, required=False)
parser.add_argument('--getFFT','-gFFT', help='Function to get FFT',default=True, required=False)
parser.add_argument('--getEnvelope','-gE', help='Function to get Envelope',default=True, required=False)
args = parser.parse_args()

#====== Frequency Bands ======
bands = [[70,150]]#,[150,250],[250,350]]

#=== Check Dates ===
def is_after(date1, date2):
    format_str = "%Y-%m-%d"  # Format for 'YYYY-MM-DD'
    d1 = datetime.strptime(date1, format_str)
    d2 = datetime.strptime(date2, format_str)
    isTrue = d1 < d2
    if isTrue:
        delta = abs((d2 - d1).days)
        months, days = divmod(delta, 30)
        print(f'Processing for {months} months and {days} days.')
    return isTrue #! HERE
assert is_after(f'{args.year}-{args.startMonth}-{args.startDay}', f'{args.year}-{args.endMonth}-{args.endDay}'), f"End date {args.endMonth}-{args.endDay} should be after start date {args.startMonth}-{args.startDay}"

#=== Data Path ===
baseLoc = f'/mnt/lfs6/exp/IceCube/{args.year}/unbiased/surface/V7_radio/radio_temp/i3_files/'
#baseLoc = f'/mnt/lfs6/exp/IceCube/{args.year}/unbiased/surface/V6/radio_temp/i3_files/'
assert os.path.exists(baseLoc), f'Path {baseLoc} does not exist'
outputBaseLoc = '/mnt/ceph1-npx/user/valeriatorres/galactic_noise/SouthPole/daily'

#====== File List ======
def list_files_between_months(start_date, end_date, directory=baseLoc):
     # List and filter files
    filtered_files = [
        f for f in glob.glob(os.path.join(directory, 'eventData_*.i3.gz'))
        if (start_date <= os.path.basename(f).split('_')[2] <= end_date)
    ]
    assert len(filtered_files) > 0, f"No files found between {start_date} and {end_date} OR the filenames format is incorrectly called."
    print(f'Found {len(filtered_files)} files between {start_date} and {end_date}')
    return filtered_files
input_files = list_files_between_months(f'{args.year}-{args.startMonth}-{args.startDay}', f'{args.year}-{args.endMonth}-{args.endDay}')
input_files = sorted(input_files)

#========= Clean Your Barn =========
def getSiderialTime(frame):
    time = astropy.time.Time(val=frame['I3EventHeader'].start_time.unix_time,format="unix",location=astropy.coordinates.EarthLocation.of_site("IceCube"))
    time_sidereal = time.sidereal_time("apparent")
    return time_sidereal

# Choosing Soft Triggers
def filterFrames(frame):
    trigger_info = frame["SurfaceFilters"]
    if trigger_info["soft_flag"]:
        return True
    return False

# Choose the Triggering Mode - Another kind of filter
def chooseTriggerMode(frame,mode):
    traceLength = frame['RadioTraceLength'].value
    traceLenDict = dict(noncascaded=1024,cascaded=1024*4,semi=1024*2)
    assert mode in traceLenDict.keys(), f"Mode {mode} not in {traceLenDict.keys()}"
    if traceLength == traceLenDict[mode]:
        return True
    return False

#=== Main Function  ===
#Subtraces method
def cutTraces(radTrace, lengthSubTraces=64, mode="rms"):
    steps = np.arange(0, len(radTrace), lengthSubTraces)
    nbSubTraces = len(radTrace) / lengthSubTraces
    temp = []
    for i in range(int(nbSubTraces)-1):
        chopped = radTrace.GetSubset(int(steps[i]), int(steps[i + 1]))
        temp.append(radcube.GetRMS(chopped))
        temp.sort()
    return temp

#This class uses the subtraces method to obtain the RMS values in each antenna and channel, and stores them in a .npz file
class GalacticBackground(icetray.I3Module):
    def __init__(self, ctx):
        icetray.I3Module.__init__(self, ctx)
        self.AddParameter('InputName', 'InputName', "InputName")
        self.AddParameter('Output', 'Output', "Output")
        self.AddParameter("ApplyInDAQ", "ApplyInDAQ", False)

    def Configure(self):
        self.inputName = self.GetParameter('InputName')
        self.output = self.GetParameter('Output')
        self.applyinDAQ = self.GetParameter("ApplyInDAQ")

        self.timeOutput = []
        self.baselineRms = []
        self.siderialTime = []

        print("... I am starting")

    def RunForOneFrame(self, frame):
        if frame.Has(self.inputName) and frame.Has("RadioTaxiTime"):
            time = frame["RadioTaxiTime"]
            time_new = np.datetime64(time.date_time).astype(datetime)
            self.timeOutput.append(time_new)

            antennaDataMap = frame[self.inputName]
            rmsTraces = []
            for iant, antkey in enumerate(antennaDataMap.keys()):
                channelMap = antennaDataMap[antkey]
                antenna_polarization_data = []
                for ichan, chkey in enumerate(channelMap.keys()):
                    fft = channelMap[ichan].GetFFTData()
                    timeSeries = fft.GetTimeSeries()
                    noises = cutTraces(timeSeries, lengthSubTraces=64)
                    rms_value = np.mean(noises[:10])
                    self.baselineRms.append(rms_value)
            # self.siderialTime.append(getSiderialTime(frame))

    def DAQ(self, frame):
        if self.applyinDAQ:
            self.RunForOneFrame(frame)

    def Physics(self, frame):
        if not self.applyinDAQ:
            self.RunForOneFrame(frame)

    def Finish(self):
        try:
            timeOutput = np.asarray(self.timeOutput)
            baselineRms = np.asarray(self.baselineRms)
            # Reshape baselineRms to have dimensions (self.counts, 3, 2)
            baselineRms_reshaped = baselineRms.reshape(-1, 3, 2)
            siderialTime = self.siderialTime
            # Save the data
            np.savez(self.output,
                    time=timeOutput,
                    # Extract data for each antenna and polarization
                    rms10 = baselineRms_reshaped[:, 0, 0],
                    rms11 = baselineRms_reshaped[:, 0, 1],
                    rms20 = baselineRms_reshaped[:, 1, 0],
                    rms21 = baselineRms_reshaped[:, 1, 1],
                    rms30 = baselineRms_reshaped[:, 2, 0],
                    rms31 = baselineRms_reshaped[:, 2, 1],
                    #siderialTime=siderialTime
                    )
        except Exception as e:
            print(f'rms error: {e}')

init_time = time.time()
logging.basicConfig(filename='bad_files.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('bad_files')

#===== Run the Horse =====
def runFile(num, file, day):
    tray = I3Tray()
    tray.AddModule("I3Reader", "reader", FileNameList=[file])
    tray.AddModule(filterFrames, "filterFrames")
    tray.AddModule(chooseTriggerMode, "chooseTriggerMode", mode=args.triggerMode)
    #Removing TAXI artifacts
    try:
        tray.Add(
            radcube.modules.RemoveTAXIArtifacts, "ArtifactRemover",
            InputName="RadioTAXIWaveform",
            OutputName="ArtifactsRemoved",
            medianOverCascades=True,
            RemoveBinSpikes=True,
            BinSpikeDeviance=int(2**12),
            RemoveNegativeBins=True,)
        tray.AddModule("I3NullSplitter","splitter",
                    SubEventStreamName="RadioEvent")
        tray.AddModule("MedianFrequencyFilter", "MedianFilter",
                    InputName="ArtifactsRemoved",
                    FilterWindowWidth=20,
                    OutputName="MedFilteredMap")
    except Exception as e:
        print(f'frame error: {e}')

    for band in bands:
        start_, end_ = band
        tray.AddModule("BandpassFilter", f"filter_{str(start_)}_{str(end_)}",
                InputName="MedFilteredMap",
                OutputName=f"FilteredMap_{str(start_)}_{str(end_)}",
                ApplyInDAQ=False,
                FilterType=radcube.eButterworth,
                ButterworthOrder=13,
                #FilterType=radcube.eBox,
                FilterLimits=[start_*I3Units.megahertz, end_*I3Units.megahertz],
                )
        tray.AddModule(GalacticBackground, f"TheGalaxyObserverDeconvolved_{start_}-{end_}",
                InputName=f"FilteredMap_{str(start_)}_{str(end_)}",
                Output=os.path.join(outputBaseLoc,f"{str(num)}_{day}_GalOscillation_Time_{args.year}_{args.startMonth}_{args.startDay}-{args.year}_{args.endMonth}_{args.endDay}_Freq_{start_}-{end_}.npz"),
                )
    tray.Execute()

for num, file_ in enumerate(input_files):
    try:
        runFile(num, file_, file_.split('/')[-1].split('_')[2])
        print(f'-- File {file_} done.')
    except Exception as e:
        print(file_)
        logger.info(file_)

print(f'-- Time elapsed: {(time.time()-init_time)/60:.2f} min.')