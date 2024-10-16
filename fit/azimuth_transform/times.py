import os
import glob

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.time import TimeMJD
import astropy.coordinates
from datetime import datetime
from astropy.coordinates import get_sun
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Galactic

from icecube.icetray import I3Tray
from icecube import icetray, dataio, dataclasses, radcube
from icecube.dataclasses import I3AntennaGeo
from icecube.icetray import I3Units
from icecube import astro
from icecube import icetray, dataio, dataclasses, radcube

import matplotlib.pyplot as plt

baseLoc = '/mnt/lfs6/exp/IceCube/2023/unbiased/surface/V7_radio/radio_temp/i3_files/'

def list_files_between_months(start_date, end_date, directory=baseLoc):
    filtered_files = [
        f for f in glob.glob(os.path.join(directory, 'eventData_*.i3.gz'))
        if (start_date <= os.path.basename(f).split('_')[2] <= end_date)
    ]
    return filtered_files
input_files = list_files_between_months('2023-01-01', '2023-01-10')

mjds = []
utc = []
for file in input_files:
    frame = dataio.I3File(file).pop_frame()
    if frame.Has("RadioTaxiTime"):
        try:
            t = np.datetime64(frame["RadioTaxiTime"].date_time.replace(tzinfo=None)).astype(datetime)
            utc.append(t)
            mjd = frame['I3EventHeader'].start_time.mod_julian_day + (frame['I3EventHeader'].start_time.mod_julian_sec / 86400.0)
            mjds.append(mjd)
        except:
            print(frame["RadioTaxiTime"])

azim_ic = np.degrees(astro.sun_dir(np.array(mjds)))[1]
print(azim_ic.shape)


#--------

data = np.load(filename, allow_pickle=True)
time = utc
time_utc = pd.to_datetime(time)
location = EarthLocation.of_site('IceCube')
time = Time(time_utc)
sun_coord = get_sun(time)
altaz_frame = AltAz(obstime=time, location=location)
sun_altaz = sun_coord.transform_to(altaz_frame)
azim_sun = sun_altaz.az.deg
print(azim_ic.shape, azim_sun.shape, time_utc.shape)


df_i = pd.DataFrame({'time':time_utc, 'astropy':azim_sun, 'icecube':azim_ic})
df_i = df_i.loc[(df_i['time'] >= '2023-01-01') & (df_i['time'] < '2023-01-10')]
df_i = df_i.sort_values(by=['time'])

df_i.to_csv('times.csv')

plt.plot(df_i['time'], df_i['astropy'], label='astropy')
plt.plot(df_i['time'], df_i['icecube'], label='icecube (using icetray mjd)')

plt.ylabel('azim')
plt.xlabel('time')
plt.xticks(rotation=30)
plt.title('Sun location')

plt.grid()
plt.legend()
plt.savefig('plot_ic.png')