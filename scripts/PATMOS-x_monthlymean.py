"""
Merge the files generated with PATMOS-x_hourlymonthlymean.py
into a single netCDF
"""

import argparse
import datetime
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

def monthly_sums(ds: xr.Dataset, var: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute the monthly sums by summing the values along the hours of the day
    
    Args:
        ds: the xarray.Dataset containing the variables `var` and `var_count` 
        var: variable for which to compute sums
    
    Returns:
        An xarray.DataArray with the sum along the hours of day and another
        xarray.DataArray with the sum of values used in the sum
    """

    a_var = ds[var]
    a_var_count = ds[f'{var}_count']

    a_var_sum = np.squeeze(
        np.nansum(
            (a_var * a_var_count),
            axis=a_var.dims.index('hour_of_day')
        )
    )

    a_var_count_sum = np.squeeze(
        np.nansum(
            a_var_count,
            axis=a_var_count.dims.index('hour_of_day')
        )
    )

    return a_var_sum, a_var_count_sum

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
        help="path to the directory where to save the compiled netCDF"
    )
    parser.add_argument(
        '--start',
        default=datetime.datetime(1982, 1, 1).date(),
        type=lambda x: datetime.datetime.strptime(x, '%Y%m').date(),
        help='for which year and month (YYYYMM) start computing the means'
    )
    parser.add_argument(
        '--end',
        default=datetime.datetime(2019, 12, 1).date(),
        type=lambda x: datetime.datetime.strptime(x, '%Y%m').date(),
        help='for which year and month (YYYYMM) end computing the means'
    )
    parser.add_argument(
        '--variables',
        nargs='+',
        default=['cloud_probability', 'cloud_fraction', 'tiwp', 'tiwp_mixed'],
        help="variables for which compute the monthly means"
    )

    args = parser.parse_args()

    # Find files
    mm_files = sorted(
        list(args.source.glob('PATMOS-x_v06-hourlymonthlymean_*.nc'))
    )

    # Filter according to start and end
    mm_files = [
        f
        for f in mm_files
        if args.start <= get_date_from_filename(f) <= args.end
    ]

    # Assert that there are as many files as expected to compute the means
    expected_mm_files_count = 12 * (args.end.year - args.start.year) + \
        args.end.month - args.start.month + 1
    assert len(mm_files) == expected_mm_files_count

    # Process the first file manually to create arrays
    ds = xr.open_dataset(mm_files[0])
    time = [ds.time.data[0].astype('datetime64[M]').astype('datetime64[ns]')]
    data = dict()
    for var in args.variables:
        var_sum, var_count_sum = monthly_sums(ds, var)
        data[var] = {'sum': var_sum[None, ...], 'count': var_count_sum[None, ...]}

    # Process the rest of the files, concatenating along the time dimension
    for f in tqdm.tqdm(mm_files[1:], dynamic_ncols=True):
        ds = xr.open_dataset(f)
        time.append(
            ds.time.data[0].astype('datetime64[M]').astype('datetime64[ns]')
        )
        for var in args.variables:
            var_sum, var_count_sum = monthly_sums(ds, var)
            data[var]['sum'] = np.concatenate(
                (data[var]['sum'], var_sum[None, ...]),
                axis=0
            )
            data[var]['count'] = np.concatenate(
                (data[var]['count'], var_count_sum[None, ...]),
                axis=0
            )
    
    # Create an xarray dataset
    ds_monthlymeans = xr.Dataset(
        data_vars={
            v: (('time', 'latitude', 'longitude'), np.divide(data[v]['sum'], data[v]['count'], out=np.full_like(data[v]['sum'], np.nan), where=data[v]['count'] > 0))
            for v in args.variables
        } | {
            f'{v}_count': (('time', 'latitude', 'longitude'), data[v]['count'])
            for v in args.variables
        },
        coords={
            'time': ('time', time),
            'latitude': ('latitude', ds.latitude.data),
            'longitude': ('longitude', ds.longitude.data)
        }
    )

    # Add units
    for var in args.variables:
        if 'units' in ds[var].attrs:
            ds_monthlymeans[var].attrs['units'] = ds[var].attrs['units']

    # Write to disk
    fname = f"PATMOS-x_v06-monthlymean_{args.start.strftime('%Y%m')}-{args.end.strftime('%Y%m')}.nc"
    ds_monthlymeans.to_netcdf(args.destination / fname)