#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2022 the i3astropy contributors
#
# SPDX-License-Identifier: BSD-2-Clause

"""Example to verify that the azimuth is defined correctly with reference to the sun.

It is obvious that the sun should be at Grid South at midnight,
Grid East at 6:00, Grid North at noon, and Grid West at 18:00
the example makes a plot to verify that coordinates are defined correctly.
"""
from i3astropy import I3Dir

from pathlib import Path
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import ICRS, get_sun, SkyCoord, EarthLocation, AltAz, Galactic
from astropy.time import Time
from astropy.units import day, deg
from matplotlib import patches, ticker

from i3astropy import I3Dir

plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Times New Roman"], 
    })

filename = 'times.csv'
data = pd.read_csv(filename)
time = data['time']
time_utc = pd.to_datetime(time).sort_values()

t = Time(time_utc, scale='utc')

sun_azimuth = get_sun(t).transform_to(I3Dir()).az.deg

plt.plot(time_utc, sun_azimuth, lw=20, label='i3astropy')
plt.plot(time_utc, (90-data['astropy'])%360, lw=10, label='astropy')
plt.plot(time_utc, data['icecube'], label='icecube (using icetray mjd)')

plt.ylabel("Azimuth")
plt.xlabel("UTC Time")
plt.xticks(rotation=30)

plt.legend()
plt.grid()
plt.savefig('plot_azim.png', dpi=150)