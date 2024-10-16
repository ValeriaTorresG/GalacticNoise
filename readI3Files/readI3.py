import argparse
import os
import sys
import glob
import time
import numpy as np
import astropy.time
import astropy.coordinates
from datetime import datetime

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, radcube
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units
from icecube import astro

# python readI3.py -stD "05" -stM "01" -enD "06" -enM "01" -y "2024"

parser = argparse.ArgumentParser(description='Read I3 files')
parser.add_argument('--startDay','-stD', help='Day',required=True)
parser.add_argument('--startMonth','-stM', help='Month',required=True)
parser.add_argument('--endDay','-enD', help='Day',required=True)
parser.add_argument('--endMonth','-enM', help='Month',required=True)
parser.add_argument('--year','-y', help='Year',required=True)
#=== Pulse Selection ===
parser.add_argument('--traceBins','-tb', type=int, help='Number of Bins in Time-trace',default=1024)
parser.add_argument('--triggerMode','-tM', help='Trigger Mode',default='noncascaded')
#=== Spectrum Extraction === # This hasn't been used yet. See analyzeFrame.py in utils. For future use.
parser.add_argument('--getTimeSeries','-gTS', help='Function to get Time Series',default=False, required=False)
parser.add_argument('--getFFT','-gFFT', help='Function to get FFT',default=True, required=False)
parser.add_argument('--getEnvelope','-gE', help='Function to get Envelope',default=True, required=False)
args = parser.parse_args()

#====== Frequency Bands ======
bands = [[90,110]]#[100,105],[115,120]]

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
outputBaseLoc = '/mnt/ceph1-npx/user/valeriatorres/galactic_noise/SouthPole'

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
# input_files = ['/mnt/lfs6/exp/IceCube/2023/unbiased/surface/V7_radio/radio_temp/i3_files/eventData_1692922207_2023-08-25_00-10-07_4095_138288.i3.gz']

#========= Clean Your Barn =========
def getSiderialTime(frame):
    time = astropy.time.Time(val=frame['I3EventHeader'].start_time.unix_time,format="unix",location=astropy.coordinates.EarthLocation.of_site("IceCube"))
    time_sidereal = time.sidereal_time("apparent")
    return time_sidereal

# Choosing Soft Triggers
def filterFrames(frame):
    trigger_info = frame["SurfaceFilters"]
    if trigger_info["soft_flag"].condition_passed:
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
    temp = sorted(temp)
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
        self.baselineRms_spread = []
        self.baselineRms_10to20 = []
        self.baselineRms_10to20_spread = []

        self.medianRMS = []

        print("... I am starting")

    def RunForOneFrame(self, frame):
        if frame.Has(self.inputName) and frame.Has("RadioTaxiTime"):
            time = frame["RadioTaxiTime"]

            time_new = np.datetime64(time.date_time.replace(tzinfo=None)).astype(datetime)
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

                    rms_value, rms_value_spread = np.mean(noises[:10]), np.std(noises[:10])
                    self.baselineRms.append(rms_value); self.baselineRms_spread.append(rms_value_spread)

                    rms_value_10to20, rms_value_10to20_spread = np.mean(noises[10:20]), np.std(noises[10:20])
                    self.baselineRms_10to20.append(rms_value_10to20); self.baselineRms_10to20_spread.append(rms_value_10to20_spread)

                    self.medianRMS.append(np.median(noises))


    def DAQ(self, frame):
        if self.applyinDAQ:
            self.RunForOneFrame(frame)

    def Physics(self, frame):
        if not self.applyinDAQ:
            self.RunForOneFrame(frame)

    def Finish(self):

        timeOutput = np.asarray(self.timeOutput)

        baselineRms = np.asarray(self.baselineRms)
        # Reshape baselineRms to have dimensions (self.counts, 3, 2)
        baselineRms_reshaped = baselineRms.reshape(-1, 3, 2)
        baselineRms_spread = np.asarray(self.baselineRms_spread)
        baselineRms_spread_reshaped = baselineRms_spread.reshape(-1, 3, 2)

        baselineRms_10to20 = np.asarray(self.baselineRms_10to20)
        baselineRms_10to20_reshaped = baselineRms_10to20.reshape(-1, 3, 2)
        baselineRms_10to20_spread = np.asarray(self.baselineRms_10to20_spread)
        baselineRms_10to20_spread_reshaped = baselineRms_10to20_spread.reshape(-1, 3, 2)

        medianRMS = np.asarray(self.medianRMS)
        medianRMS_reshaped = medianRMS.reshape(-1, 3, 2)

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

                rms10_spread = baselineRms_spread_reshaped[:, 0, 0],
                rms11_spread = baselineRms_spread_reshaped[:, 0, 1],
                rms20_spread = baselineRms_spread_reshaped[:, 1, 0],
                rms21_spread = baselineRms_spread_reshaped[:, 1, 1],
                rms30_spread = baselineRms_spread_reshaped[:, 2, 0],
                rms31_spread = baselineRms_spread_reshaped[:, 2, 1],

                rms10_10to20 = baselineRms_10to20_reshaped[:, 0, 0],
                rms11_10to20 = baselineRms_10to20_reshaped[:, 0, 1],
                rms20_10to20 = baselineRms_10to20_reshaped[:, 1, 0],
                rms21_10to20 = baselineRms_10to20_reshaped[:, 1, 1],
                rms30_10to20 = baselineRms_10to20_reshaped[:, 2, 0],
                rms31_10to20 = baselineRms_10to20_reshaped[:, 2, 1],

                rms10_10to20_spread = baselineRms_10to20_spread_reshaped[:, 0, 0],
                rms11_10to20_spread = baselineRms_10to20_spread_reshaped[:, 0, 1],
                rms20_10to20_spread = baselineRms_10to20_spread_reshaped[:, 1, 0],
                rms21_10to20_spread = baselineRms_10to20_spread_reshaped[:, 1, 1],
                rms30_10to20_spread = baselineRms_10to20_spread_reshaped[:, 2, 0],
                rms31_10to20_spread = baselineRms_10to20_spread_reshaped[:, 2, 1],

                medianRMS10 = medianRMS_reshaped[:, 0, 0],
                medianRMS11 = medianRMS_reshaped[:, 0, 1],
                medianRMS20 = medianRMS_reshaped[:, 1, 0],
                medianRMS21 = medianRMS_reshaped[:, 1, 1],
                medianRMS30 = medianRMS_reshaped[:, 2, 0],
                medianRMS31 = medianRMS_reshaped[:, 2, 1],
                )


#===== Run the Horse =====
init_time = time.time()
tray = I3Tray()
tray.AddModule("I3Reader", "reader", FileNameList=input_files)
tray.AddModule(filterFrames, "filterFrames", Streams = [icetray.I3Frame.DAQ])
tray.AddModule(chooseTriggerMode, "chooseTriggerMode", mode=args.triggerMode, Streams = [icetray.I3Frame.DAQ])
#Removing TAXI artifacts
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
            #    Output=f"GalOscillation_Time_{args.year}.npz",
               Output=os.path.join(outputBaseLoc,f"GalOscillation_Time_{args.year}_{args.startMonth}_{args.startDay}-{args.year}_{args.endMonth}_{args.endDay}_Freq_{start_}-{end_}.npz"),
               )
tray.Execute()
print(f'-- Time elapsed: {(time.time()-init_time)/60:.2f} min.')