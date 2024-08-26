import argparse
import os
import sys
import glob

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, radcube
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units


parser = argparse.ArgumentParser(description='Read I3 files')
parser.add_argument('input_files', nargs='+', help='Input files')
parser.add_argument('--year','-y', type=int, help='Year',required=True)
parser.add_argument('--startMonth','-stM', type=int, help='Month',required=True)
parser.add_argument('--startDay','-stD', type=int, help='Day',required=True)
parser.add_argument('--WaveLength','-wl', type=int, help='WaveLength',default=1024)
args = parser.parse_args()

baseLoc = f'/mnt/lfs6/exp/IceCube/{args.year}/unbiased/surface/V7_radio/radio_temp/i3_files/'
assert os.path.exists(baseLoc), f'Path {baseLoc} does not exist'

tray = dataio.I3Tray()
