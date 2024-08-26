#!/usr/bin/env python3

#This is an example script which runs the star interpolator module
#Choose a simulation(s) to run on and a gcd file to use as the interpolation targets and it outputs an I3File
#Run as: ./InterpolateStar.py /data/sim/IceCubeUpgrade/CosmicRay/Radio/coreas/data/continuous/star-pattern/proton/lgE_16.5/sin2_0.7/000000

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, radcube

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, nargs='+', default=[], help='List of CoREAS simulation directories')
parser.add_argument('--gcd', type=str, default="/data/user/acoleman/datasets/gcd-files/GCD-AntennaSurvey_recent.i3.gz", help='The location of the gcd file on which you want to interpolate the shower location')
parser.add_argument('--output', type=str, default="MyOutput.i3.gz", help='File to write to')
args = parser.parse_args()

tray = I3Tray()

tray.AddModule("I3InfiniteSource", "source",
              prefix = args.gcd,
              stream = icetray.I3Frame.DAQ)

tray.AddModule(radcube.modules.SimulateFromStar, "SimulateFromStar",
               DirectoryList = args.input,
               NThrows = 3,                       #How many Q-frames to make per CoREAS file
               RandomCoreCenter = [-250, -50],    #Location about which the core is randomly chosen
               CoreRadius = 10,                   #Radius about RandomCoreCenter that the core is chosen
               RNGSeed = 477,                     #Random number generator seed value
              )

tray.AddModule("I3Writer","i3writer",
               filename=args.output,
               streams=[icetray.I3Frame.Geometry, icetray.I3Frame.DetectorStatus,
                        icetray.I3Frame.Calibration, icetray.I3Frame.DAQ,
                        icetray.I3Frame.Physics]
               )

tray.Execute()