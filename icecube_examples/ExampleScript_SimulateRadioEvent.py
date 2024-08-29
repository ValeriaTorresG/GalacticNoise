#!/bin/env python3

#This is the standard simulation script for reading in a non-star-shaped antenna simulation. An output file is written out
#Run as: python ExampleScript_SimulateRadioEvent.py /data/sim/IceCubeUpgrade/CosmicRay/Radio/coreas/data/discrete/proton/lgE_17.5/Zen_51/000000

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', dest='outfile', type=str, default='simulated-radio-event.i3.gz', help='Name of the output I3File at the end of the sim')
parser.add_argument('input', type=str, help='Input data file.')
args = parser.parse_args()


from icecube.icetray import I3Tray
from icecube import icetray, radcube, dataio
from icecube.radcube import defaults

print("Howdy!")
print("Reading in CoREAS file(s): " + args.input)
print("Will make I3 file: " + args.outfile)

tray = I3Tray()

#These lines create basic response services. See radcube.defaults for more
#The functions place them in the frame and give you back the name
antennaResponse = defaults.CreateDefaultAntennaResponse(tray)
electronicsResponse = defaults.CreateDefaultElectronicsResponse(tray)
randomService = defaults.CreateDefaultRandomService(tray)

#This service reads in the CoREAS event, adds noise, convolves it with the
#antenna/electronics response, "digitizes" it, and puts it in the frame.
tray.AddSegment(radcube.segments.RadioInjection, 'RadioInjection',
                 InputFiles = [args.input],
                 MakeGCD = True,
                 AntennaServiceName = antennaResponse,
                 ElectronicServiceName = electronicsResponse,
                 RandomServiceName = randomService,
                 OutputName = "InjectedVoltageMap"
                )

#This module will delete things from the frame before it saves to reduce data output
tray.AddModule('Delete', 'Delete',
               KeyStarts=['RadioInjection_'], #Deletes intermitant steps in injection
               Keys=['GeneratedNoiseMap',\
                     'CoREASEFieldMap'], #Deltes the pure noise signal and original EField
              )

tray.AddModule("I3Writer","i3writer",
               filename=args.outfile,
               )


tray.Execute()