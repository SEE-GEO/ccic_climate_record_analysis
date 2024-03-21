import numpy as np
import xarray as xr
from pathlib import Path
from pansat.products.satellite.cloud_sat import l2c_ice, l2b_cldclass
from datetime import timedelta, datetime
from scipy.interpolate import interpn
import ccic

def get_time_range(times):
    """
    get first and last times for CloudSat granule as datetime objects on the form yyyy-mm-dd hh:mm:ss
    
    times: numpy ndarray with datetime64[ns]
    """
    start = np.datetime_as_string(np.min(times), unit='s').replace('T', ' ')
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    stop = np.datetime_as_string(np.max(times), unit='s').replace('T', ' ')
    stop = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S')
    #start = np.min(times).tolist()
    #stop = np.max(times).tolist()
    return start, stop

def get_local_time_grid(tmin, tmax, dt):
    """
    generates a local part of a grid of datetimes with values at hours
    00+n*dt that covers tmin and tmax.
 
    tmin:   datetime object specifying the smalles time that the local 
            grid should cover
    tmax:   datetime object specifying the largest time that the local  
            grid should cover
    dt:     int specifying the timestep of the grid
    """
    # get start time
    h = (tmin.hour // dt)*dt
    t0 = tmin.replace(hour=h, minute=0, second=0)
    # get end time
    h = ((tmax.hour + dt) // dt)*dt
    if h > 23:
        tmax += timedelta(days=1)
        t1 = tmax.replace(hour=0, minute=0, second=0)
    else:
        t1 = tmax.replace(hour=h, minute=0, second=0)
    dt = timedelta(hours=dt)
    return np.arange(t0, t1+dt, dt).tolist()

def load_dardar(filepath : Path):

    # 'time' and 'height' are dimensions
    variables = [
        'latitude',
        'longitude',
        'iwc',
        'instrument_flag',
        'DARMASK_Simplified_Categorization',
    ]

    latlim = (-70, 69)  #  latitude limits for Gridsat

    with xr.open_dataset(filepath) as ds:
        # keep selected variables
        ds = ds[variables]
        # filter on latitudes
        idxs = ds.latitude > latlim[0]
        ds = ds.isel(time=idxs)
        idxs = ds.latitude < latlim[1]
        ds = ds.isel(time=idxs)
        # reverse 'height' coordinate to go from -1020 to 25080
        ds = ds.reindex(height=ds.height[::-1]) 
        return ds
    
def load_2cice(filepath : Path):

    # 'rays' and 'bins' are dimensions
    variables = [
        'latitude',
        'longitude',
        'height',
        'iwc',
        'iwp',
        'time_since_start',
    ]

    latlim = (-70, 69)  #  latitude limits for Gridsat

    ds = l2c_ice.open(filepath)
    ds = ds.reset_coords()
    # keep selected variables
    ds = ds[variables]
    # filter on latitudes
    idxs = ds.latitude > latlim[0]
    ds = ds.isel(rays=idxs)
    idxs = ds.latitude < latlim[1]
    ds = ds.isel(rays=idxs)
    # reverse 'bins' coordinate to make height go from low to high values
    ds = ds.reindex(bins=ds.bins[::-1])

    return ds

def load_cldclass(filepath : Path):

    # 'rays' and 'bins' are dimensions
    variables = [
        'latitude',
        'height',
        'surface_elevation',
        'cloud_class',
        'cloud_class_flag',
    ]

    latlim = (-70, 69)  #  latitude limits for Gridsat

    ds = l2b_cldclass.open(filepath)
    ds = ds.reset_coords()
    # keep selected variables
    ds = ds[variables]
    # filter on latitudes
    idxs = ds.latitude > latlim[0]
    ds = ds.isel(rays=idxs)
    idxs = ds.latitude < latlim[1]
    ds = ds.isel(rays=idxs)
    # reverse 'bins' coordinate to make height go from low to high values
    ds = ds.reindex(bins=ds.bins[::-1])

    return ds

def load_ccic(filepath : Path):

    variables = [
        'cloud_prob_2d',
        'tiwp',
    ]
    with xr.open_dataset(filepath, engine='zarr') as ds:
        # Filter on latitude
        # NOTE latitudes in the cpcir dataset goes from 60 to -60 deg
        # ds = ds.sel(
        #     latitude=slice(latlim[1], latlim[0]),
        # )
        # # Split dataset into separate utc times
        # data = [ds.isel(time=0), ds.isel(time=1)]
        # keep selected variables
        ds = ds[variables]
        return ds
    
def load_ccic_grid(local_time_grid, ccic_datadir):
    """
    """
    dsets = []
    for t in local_time_grid:
        timestamp = t.strftime("%Y%m%d%H%M")
        file_to_load = f'{timestamp[0:4]}/ccic_gridsat_{timestamp}.zarr'
        filepath = Path(ccic_datadir, file_to_load)
        # Load data
        dsets.append(load_ccic(filepath))
    return xr.concat(dsets, dim='time')

def load_era5(filepath : Path):

    variables = [
        'tiwp',
        'tcc',
    ]
    with xr.open_dataset(filepath) as ds:
        # add column snow and column ice to get total ice water path
        ds['tiwp'] = ds.tcsw + ds.tciw
        ds = ds[variables]
        return ds

def load_era5_grid(local_time_grid, datadir):
    """
    """
    dsets = []
    for t in local_time_grid:
        timestamp = t.strftime("%Y%m%d%H")
        file_to_load = f'reanalysis-era5-single-levels_{timestamp}_total_column_snow_water-total_cloud_cover-total_column_cloud_ice_water-90-90--180-180.nc'
        filepath = Path(datadir, file_to_load)
        # Load data
        dsets.append(load_era5(filepath))
    return xr.concat(dsets, dim='time')


def process_dardar(ds):

    # calculate iwp (iwc is in kg/m^3, height is in m, dz = 60 m)
    iwp = np.trapz(ds.iwc.values, ds.height.values, axis=1)
    
    # get cloud mask 
    cm = ds.DARMASK_Simplified_Categorization.values
    # remap cloud classes to 4 cloud mask flags
    cm[np.isin(cm, [-2, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15])] = 1
    cm[np.isin(cm, [6, 7, 8])] = 0
    # count cloud mask categories
    cldf = profile_cloud_fraction(cm)

    data = {
        'lat'   :   ds.latitude.values,
        'lon'   :   ds.longitude.values,
        'time'  :   ds.time.values,
        'iwp'   :   iwp,
        'cm'    :   cldf,
    }
    return data


def process_2cice(ds, cldclass_ds):

    # get UTC times
    #timeutc = utc_times_2cice(ds)   # array with datetime64[s]

    # calculate iwp (iwc is in g/m^3, height is in m, dz = 240 m)
    #iwp_i = np.trapz(ds.iwc.values/1000, ds.height.values, axis=1)
    iwp = ds.iwp.values/1000   

    # get cloud mask   
    cm = get_cldclass_cloud_mask(cldclass_ds)
    # count cloud mask categories
    cldf = profile_cloud_fraction(cm)
    
    data = {
        'lat'   :   ds.latitude.values,
        'lon'   :   ds.longitude.values,
        #'time'  :   timeutc,
        'iwp'   :   iwp,
        'cm'    :   cldf,
    }
    return data

def get_cldclass_cloud_mask(ds):
    """
    ds: dataset with 2B-CLDCLASS data
    """
    sf_elevation = ds.surface_elevation.values
    sf_elevation[sf_elevation == -9999] = 0

    # bins classified as some cloud --> 1
    cm = ds.cloud_class.values
    cm[cm > 0] = 1
    # bins with unvalid height --> -9 
    cm[ds.height.values == -9999] = -9
    # bins with unsuccessfully determined cloud class --> -9
    cm[ds.cloud_class_flag.values == 0] = -9
    # bins below surface --> -1
    mask = (ds.height.values < ds.surface_elevation.values[:, np.newaxis]) & (ds.height.values > -9999)
    cm[mask] = -1
    return cm

def profile_cloud_fraction(cm_2d):
    
    # Count cloudy bins (count 1)
    ncloudy = np.count_nonzero(cm_2d == 1, axis=1)
    # Count bins above the surface, i.e., valid bins (all bins - count -1)
    nvalid = cm_2d.shape[1]-np.count_nonzero(cm_2d == -1, axis=1)
    # Calculate profile cloud fraction
    cld_fraction = ncloudy/nvalid
    # put profiles with all bins = -9 to nan
    cld_fraction[np.where((cm_2d == -9).all(axis=1))] = np.nan
    return cld_fraction

def interpolate_data(data, times, lats, lons, variables=['tiwp']):
    """
    data: xarray dataset with dimensions 
            'time' datetime64[ns], 'latitude' float, 'longitude' float

    times: Cloudsat coordinates (datetime64[ns])
    lats:  Cloutsat coortinates (float)
    lons:  Cloutsat coortinates (float)
    """
    
    gridsat_coords = np.stack(
        [times.astype(float), lats, lons],
        axis=1
    )
    data_grid = (
        data.time.values.astype(float), 
        data.latitude.values, 
        data.longitude.values,
    )
    result = {}
    for variable in variables:
        interpolated = interpn(
            data_grid,
            data[variable].values,
            gridsat_coords,
            method='nearest',
            bounds_error=False,
            fill_value=np.nan,
        )
        result[variable] = interpolated
    return result



