"""
Compute monthly means from the hourly monthly means
"""

import argparse
import datetime
import functools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm
import xarray as xr

def get_date_from_filename(p: Path) -> datetime.date:
    """
    Extract the year and month from the filename

    Args:
        p: path to the hourly monthly mean file
    
    Returns:
        The string YYYYMM extracted from the filename
    """
    return datetime.datetime.strptime(
        p.stem.split('_')[-1],
        '%Y%m'
    ).date()

def process_hmm_file(file_hmm: Path, variables: list[str],
                     destination: Path) -> None:
    """
    Compute monthly means from a hourly monthly mean file, saving it to disk.

    Args:
        file_hmm: path to the hourly monthly mean file
        variables: for which variables compute the means
        destination: where to write the monthly mean file

    Notes:
        The number of data points in each pixel is taken into account.
    """
    ds_hourlymonthlymean = xr.open_dataset(file_hmm)
    data = dict.fromkeys(variables)
    for var in variables:
        data[var] = {
            # Weighted mean by number of profiles in each hour, i.e.
            # revert the aggregation by hour of day
            'mean': ds_hourlymonthlymean[var].weighted(
                ds_hourlymonthlymean[f'{var}_count']).mean(
                    dim='hour_of_day', skipna=True
                ),
            'count': ds_hourlymonthlymean[f'{var}_count'].sum(
                dim='hour_of_day', skipna=True
            )
        }
    dims = ('time', 'latitude', 'longitude')
    ds_monthlymean = xr.Dataset(
        data_vars={
            var: (dims, data[var]['mean'].data)
            for var in variables
        } | {
            f"{var}_count": (dims, data[var]['count'].data)
            for var in variables
        },
        coords={
            'time': (
                'time',
                ds_hourlymonthlymean.time.data.astype(
                    'datetime64[M]'
                    ).astype('datetime64[ns]')
            ),
            'latitude': (
                'latitude',
                ds_hourlymonthlymean.latitude.data
                ),
            'longitude': (
                'longitude',
                ds_hourlymonthlymean.longitude.data
                )
        }
    )

    # Add units
    for var in variables:
        if 'units' in ds_hourlymonthlymean[var].attrs:
            units = ds_hourlymonthlymean[var].attrs['units']
            ds_monthlymean[var].attrs['units'] = units

    # Write to disk
    month = get_date_from_filename(file_hmm).strftime('%Y%m')
    file_mm = destination / f"PATMOS-x_v06-monthlymean_{month}.nc"
    ds_monthlymean.to_netcdf(
        file_mm,
        # using this encoding the files can be about 3x smaller
        encoding={
            var: {'zlib': True, 'complevel': 9}
            for var in ds_monthlymean
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        required=True,
        type=Path,
        help=(
            "path to the location of the netCDFs "
            "generated with PATMOS-x_hourlymonthlymean.py"
        )
    )
    parser.add_argument(
        '--destination',
        required=True,
        type=Path,
        help="path to the directory where to save the new netCDFs"
    )
    parser.add_argument(
        '--variables',
        nargs='+',
        default=['cloud_probability', 'cloud_fraction', 'tiwp', 'tiwp_mixed'],
        help="variables for which compute the monthly means"
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=16,
        help="number of parallel processes to use"
    )

    args = parser.parse_args()

    # Find files
    hmm_files = sorted(
        list(args.source.glob('PATMOS-x_v06-hourlymonthlymean_*.nc'))
    )

    # To make it work with tqdm and multiprocessing.Pool
    process_hmm_file_partial = functools.partial(
        process_hmm_file,
        variables=args.variables,
        destination=args.destination
    )

    with Pool(args.processes) as pool:
        list(
            tqdm.tqdm(
                pool.imap(process_hmm_file_partial, hmm_files),
                dynamic_ncols=True,
                total=len(hmm_files)
            )
        )