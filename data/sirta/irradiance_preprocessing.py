"""
Creating data set
Need to add - my dataset directory
"""
### Irradiance forecast from sky images

import pandas as pd
import re
import os

#DATADIR = 'C:\\Users\\quent\\Documents\\~Cambridge\\~Mphil\\Research project\\Datasets\\Sirta\\Irradiance\\radiometry\\solys2'
DATADIR = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Sirta/Irradiance/solys2_bis/radflux/"

#DATADIR_DESTINATION = 'C:\\Users\\quent\\Documents\\~Cambridge\\~Mphil\\Research project\\Datasets\\Sirta\\Irradiance\\radiometry\\solys2\\preprocessed_irradiance_data\\'
DATADIR_DESTINATION = "/home/quentin/Documents/Cambridge/Research_project/Datasets/Sirta/Irradiance/solys2_bis/preprocessed_irradiance_data/"


names=['day', 'direct_solar_flux', 'diffuse_solar_flux', 'global_solar_flux', 'downwelling_solar_flux', 'minimum_direct_solar_flux',
       'minimum_diffuse_solar_flux', 'minimum_global_solar_flux', 'minimum_downwelling_IR_flux', 'maximum_direct_solar_flux',
       'maximum_diffuse_solar_flux', 'maximum_global_solar_flux', 'maximum_downwelling_IR_flux', 'standard_deviation_direct_solar_flux',
       'standard_deviation_diffuse_solar_flux', 'standard_deviation_global_folar_flux', 'standard_deviation_downwelling_IR_flux',
       'number_of_values', 'pyrgeometer_temperature', 'solar_zenith_angle', 'solar_azimuthal_angle']

def create_irradiance_dataset():
    for m in range(1,13):
        if m<=9: M='0{}'.format(m)
        else: M='{}'.format(m)

        for d in range(1,32):
            if d<=9: D='0{}'.format(d)
            else: D='{}'.format(d)

            file_name = '2014/radflux_1a_Lz2M1minIsolys2PrayDp_v01_2014{}{}_000000_1440.txt'.format(M,D)
            csv_file_name = '2014/solys2_radflux_2014{}{}.csv'.format(M,D)
            path = os.path.join(DATADIR, file_name)
            destination_path = os.path.join(DATADIR_DESTINATION, csv_file_name)

            if os.path.isfile(path)==True:

                df = pd.read_csv(path, skiprows=44, header=None)#, infer_datetime_format =True)#, sep='delimiter')#, names=names)

                for row in df.index:
                    df[0][row]=re.sub(' +', ' ', df[0][row]) #or ReGex

                df = df[0].str.split(' ',expand=True)

                df[0] = df[0].str.split('T',expand=True)[0]

                df.columns=names

                for row in df.index:
                    df['day'][row]=re.sub('2014-', '', df['day'][row])

                df.insert(loc=1, column = 'month', value=df['day'].str.split('-',expand=True)[0])
                df['day'] = df['day'].str.split('-',expand=True)[1]

                for row in df.index:
                    for col in df.columns:
                        df[col][row]=float(df[col][row])

                for row in df.index:
                    df['day'][row]=df['day'][row]+30*(df['month'][row]-1) #technically not exact

                df.to_csv(destination_path)

create_irradiance_dataset()