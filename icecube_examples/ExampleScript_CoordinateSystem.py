#!/bin/env python3

# Example on how to use the coordinate system transformations

import numpy as np
from icecube import radcube, dataclasses
from icecube.icetray import I3Units
from icecube.dataclasses import I3Constants

# Begin with an air shower
airShower = dataclasses.I3Particle()
# Put it somewhere near the surface
airShower.pos = dataclasses.I3Position(30, 40, 1950)
# It is going south-west
airShower.dir = dataclasses.I3Direction(30 * I3Units.degree, 60 * I3Units.degree)

# Consider a few positions
pos1 = dataclasses.I3Position(0, 0, 0)
pos2 = dataclasses.I3Position(0, 0, 1950)
pos3 = dataclasses.I3Position(30, 40, 1950)

# Option 1 is to get the location in another coordinate system with the shower axis going through
# x=0, y=0 in the new coordinate system. To do this, just subtract off the location of the core before
# the conversion (i.e. rotation) is performed. This is what one typically will want to do!
print("========= Option 1 ==========")
pos1InSC = radcube.GetShowerFromIC(pos1 - airShower.pos, airShower.dir)
pos1InVxB = radcube.GetMagneticFromIC(pos1 - airShower.pos, airShower.dir)
print("Pos 1 at", pos1, "is at", pos1InSC, "in shower coords")
print("Pos 1 at", pos1, "is at", pos1InVxB, "in VxB coords")
print("Pos 1 has a time delay of", pos1InSC.z / I3Constants.c, "ns in shower coords")
print("Pos 1 has a time delay of", pos1InVxB.z / I3Constants.c, "ns in shower coords")
print("")

pos2InSC = radcube.GetShowerFromIC(pos2 - airShower.pos, airShower.dir)
pos2InVxB = radcube.GetMagneticFromIC(pos2 - airShower.pos, airShower.dir)
print("Pos 2 at", pos2, "is at", pos2InSC, "in shower coords")
print("Pos 2 at", pos2, "is at", pos2InVxB, "in VxB coords")
print("Pos 2 has a time delay of", pos2InSC.z / I3Constants.c, "ns in shower coords")
print("Pos 2 has a time delay of", pos2InVxB.z / I3Constants.c, "ns in shower coords")
print("")


pos3InSC = radcube.GetShowerFromIC(pos3 - airShower.pos, airShower.dir)
pos3InVxB = radcube.GetMagneticFromIC(pos3 - airShower.pos, airShower.dir)
print("Pos 3 at", pos3, "is at", pos3InSC, "in shower coords")
print("Pos 3 at", pos3, "is at", pos3InVxB, "in VxB coords")
print("Pos 3 has a time delay of", pos3InSC.z / I3Constants.c, "ns in shower coords")
print("Pos 3 has a time delay of", pos3InVxB.z / I3Constants.c, "ns in shower coords")
print("")

# Option 2: Get the location of pos 1,2,3 in a coordinate system using a rotation about the
# standard IC origin AKA (0,0,0) in IC. Note that this one is less commonly used!
print("\n========= Option 2 ==========")
pos1InSC = radcube.GetShowerFromIC(pos1, airShower.dir)
pos1InVxB = radcube.GetMagneticFromIC(pos1, airShower.dir)
print("Pos 1 at", pos1, "is at", pos1InSC, "in shower coords")
print("Pos 1 at", pos1, "is at", pos1InVxB, "in VxB coords")
print("")

print("Pos 2 at", pos2, "is at", radcube.GetShowerFromIC(pos2, airShower.dir), "in shower coords")
print("Pos 2 at", pos2, "is at", radcube.GetMagneticFromIC(pos2, airShower.dir), "in VxB coords")
print("")
print("")

print("Pos 3 at", pos3, "is at", radcube.GetShowerFromIC(pos3, airShower.dir), "in shower coords")
print("Pos 3 at", pos3, "is at", radcube.GetMagneticFromIC(pos3, airShower.dir), "in VxB coords")