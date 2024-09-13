
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

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, radcube
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units
from icecube import astro

# python get_i3_files.py -stD "05" -stM "08" -enD "17" -enM "09" -y "2023"

parser = argparse.ArgumentParser(description='Read I3 files')
parser.add_argument('--startDay','-stD', help='Day',required=True)
parser.add_argument('--startMonth','-stM', help='Month',required=True)
parser.add_argument('--endDay','-enD', help='Day',required=True)
parser.add_argument('--endMonth','-enM', help='Month',required=True)
parser.add_argument('--year','-y', help='Year',required=True)
#=== Pulse Selection ===
parser.add_argument('--traceBins','-tb', type=int, help='Number of Bins in Time-trace',default=1024)
parser.add_argument('--triggerMode','-tM', help='Trigger Mode',default='noncascaded')
args = parser.parse_args()

#====== Frequency Bands ======
bands = [[70,150]]

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
    return isTrue
assert is_after(f'{args.year}-{args.startMonth}-{args.startDay}', f'{args.year}-{args.endMonth}-{args.endDay}'), f"End date {args.endMonth}-{args.endDay} should be after start date {args.startMonth}-{args.startDay}"

#=== Data Path ===
baseLoc = f'/mnt/lfs6/exp/IceCube/{args.year}/unbiased/surface/V7_radio/radio_temp/i3_files/'
assert os.path.exists(baseLoc), f'Path {baseLoc} does not exist'
outputBaseLoc = '/mnt/ceph1-npx/user/valeriatorres/galactic_noise/SouthPole/i3_files/'

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

#========= Clean Your Barn =========
def getSiderialTime(frame):
    time = astropy.time.Time(val=frame['I3EventHeader'].start_time.unix_time, format="unix", location=astropy.coordinates.EarthLocation.of_site("IceCube"))
    time_sidereal = time.sidereal_time("apparent")
    return time_sidereal

# Choosing Soft Triggers
def filterFrames(frame):
    trigger_info = frame["SurfaceFilters"]
    if trigger_info["soft_flag"]:
        return True
    return False

# Choose the Triggering Mode - Another kind of filter
def chooseTriggerMode(frame, mode):
    traceLength = frame['RadioTraceLength'].value
    traceLenDict = dict(noncascaded=1024, cascaded=1024*4, semi=1024*2)
    assert mode in traceLenDict.keys(), f"Mode {mode} not in {traceLenDict.keys()}"
    if traceLength == traceLenDict[mode]:
        return True
    return False

#===== Run the Horse =====
init_time = time.time()

for input_file in input_files:
    try:
        # Extract the date from the file name
        file_date = os.path.basename(input_file).split('_')[2]
        output_file = os.path.join(outputBaseLoc, f'processed_{file_date}.i3.gz')

        tray = I3Tray()
        tray.AddModule("I3Reader", "reader", Filename=input_file)
        tray.AddModule(filterFrames, "filterFrames")
        tray.AddModule(chooseTriggerMode, "chooseTriggerMode", mode=args.triggerMode)

        # Removing TAXI artifacts
        tray.Add(
            radcube.modules.RemoveTAXIArtifacts, "ArtifactRemover",
            InputName="RadioTAXIWaveform",
            OutputName="ArtifactsRemoved",
            medianOverCascades=True,
            RemoveBinSpikes=True,
            BinSpikeDeviance=int(2**12),
            RemoveNegativeBins=True,)

        tray.AddModule("I3NullSplitter", "splitter",
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
                        FilterLimits=[start_ * I3Units.megahertz, end_ * I3Units.megahertz],)

        # Add the I3Writer module to save the processed data
        tray.AddModule("I3Writer", "writer",
                    Filename=output_file,
                    Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])

        tray.Execute()
        tray.Finish()
    
    except Exception as e:
        print(e, input_file)

print(f'-- Time elapsed: {(time.time()-init_time)/60:.2f} min.')