import numpy as np
import xarray as xr
import os
import glob
from pathlib import Path
import multiprocessing

import data_preprocessing as dp

def get_granules(year, month, cloudsat_paths):
    granules = {}
    for product in cloudsat_paths:
        files = glob.glob(f'{cloudsat_paths[product]}/{year}/{month:02}/*/*')
        for f in files:
            file = Path(f)
            filename = file.name
            if filename.endswith('.hdf'):
                tstamp = filename.split('_')[0]
                id = filename.split('_')[1]
            elif filename[0:6] == 'DARDAR':
                tstamp = filename.split('_')[1]
                id = filename.split('_')[2]
            else:
                id = 0
                tstamp = 0
                print('No match for ', filename)

            if id not in granules.keys():
                granules[id] = {'files' : [file], 'tstamp' : tstamp}
            else:
                granules[id]['files'].append(file)
    return granules

def process_granule(id, granules, paths):
    # no local path needed
    # get basename from the full path
    output_dir = paths['output_dir']
    ccic_datadir = paths['ccic_datadir']
    
    if len(granules[id]['files']) == 3:    
        
        tstamp = granules[id]['tstamp']
        # Load CloudSat data
        allfiles_ok = 0
        for f in granules[id]['files']:
            filename = f.name 
            if filename.startswith('DARDAR'):
                dardar = dp.load_dardar(f)
                allfiles_ok += 1
            elif filename.startswith(f'{tstamp}_{id}_CS_2C-ICE'):
                ice2c = dp.load_2cice(f)
                allfiles_ok += 1
            elif filename.startswith(f'{tstamp}_{id}_CS_2B-CLDCLASS'):
                cldclass = dp.load_cldclass(f)
                allfiles_ok += 1
            else:
                print(f'File not recognized: {filename}')
        if allfiles_ok == 3:       
            # Process CloudSat data
            dardar = dp.process_dardar(dardar)
            ice2c = dp.process_2cice(ice2c, cldclass)
            
            # get first and last times for CloudSat granule as a datetime object
            start, stop = dp.get_time_range(dardar['time'])
            # Generate local ccic time grid
            dt = 3 # 3h timesteps for Gridsat
            ccic_t_grid = dp.get_local_time_grid(start, stop, dt)
            
            # Load CCIC data 
            try:
                ccic_ds = dp.load_ccic_grid(ccic_t_grid, ccic_datadir)
            except Exception as e:
                ccic_ds = None
                print(f'ERROR loading ccic data: ', e)

            # get CloudSat coordinates
            times = dardar['time']
            lats = dardar['lat']
            lons = dardar['lon']
            if ccic_ds is not None:
                # Interpolate CCIC data
                result = dp.interpolate_data(
                    ccic_ds,
                    times,
                    lats, 
                    lons,
                    list(ccic_ds.keys())
                    #batch_size=100,
                )
            else:
                # set values to nan
                result = {
                    'tiwp'           :   np.empty(lats.shape)*np.nan,
                    'cloud_prob_2d'  :   np.empty(lats.shape)*np.nan,
                }
            # Create dataset to save
            ds = xr.Dataset(
                data_vars=dict(
                    iwp_ccic=(['rays'], result['tiwp']),
                    iwp_2cice=(['rays'], ice2c['iwp']),
                    iwp_dardar=(['rays'], dardar['iwp']),
                    cloud_mask_ccic=(['rays'], result['cloud_prob_2d']),
                    cloud_mask_2cice=(['rays'], ice2c['cm']),
                    cloud_mask_dardar=(['rays'], dardar['cm']),
                ),
                coords=dict(
                    rays=np.arange(lons.size),
                    longitude=(['rays'], lons),
                    latitude=(['rays'], lats),
                    time=(['rays'], times),
                ),
                attrs=dict(
                    cloudsat_granule=id,
                    cloudsat_timestamp=tstamp,
                ),
            )
            # Name of output file
            outfile = f'ccicgridsat_dardar_2cice_{tstamp}_{id}.nc'
            outpath = os.path.join(output_dir, outfile)
            ds.to_netcdf(outpath)
            message = f'Processesing done for granule {id}'
        else:
            message = f'Maybe wierd filepaths for granule {id}'
    else:
        message = f'Incomplete data for granule {id}'
    return message

def run(args):
    id = args['id']
    granules = args['granules']
    paths = args['paths']
    return process_granule(id, granules, paths)


if __name__ == "__main__":

    # Path to CCIC data
    datadir_ccic = '/home/hallborn/mnt/sun/ccic/gridsat/'
    # Paths to CloudSat data
    datadir_dardar = '/data/s6/L2_ext/DARDAR/dardar3.1/'
    datadir_2cice = '/data/s6/L2_ext/2C-ICE/'
    datadir_cldclass = '/data/s6/L2_ext/2B-CLDCLASS/'

    # path to store output data
    output_dir = '/scratch/ccic_climate_record/cloudsat_collocations'
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        'output_dir'    : output_dir,
        'ccic_datadir'  : datadir_ccic,
    }

    cloudsat_paths = {
        'dardar'    :   datadir_dardar,
        '2cice'     :   datadir_2cice,
        'cldclass'  :   datadir_cldclass,
    }

    years = [2017]
    for yr in years:

        for month in range(1, 2):
            
            # Get dict with existing granules
            granules = get_granules(yr, month, cloudsat_paths)
            
            arguments = [{'id' : id, 'granules' : granules, 'paths' : paths} for id in granules]
            
            # the number of CPU cores to use
            nprocesses = 8 #multiprocessing.cpu_count() 
            count = 1
            with multiprocessing.Pool(processes=nprocesses) as pool:
                for id_message in pool.imap(run, arguments):
                    print(f'{count}/{len(arguments)} {id_message}') 
                    count += 1

            print(f'Finnished processing {yr}, month {str(month).zfill(2)}')
