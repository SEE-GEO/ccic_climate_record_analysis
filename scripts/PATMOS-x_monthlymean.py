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

def monthly_means(a: xr.DataArray) -> xr.DataArray:
    """Compute the monthly mean by averaging the hourly monthly means
    
    Args:
        a: the xarray.DataArray for which to compute the average
    
    Returns:
        An xarray.DataArray with the monthly mean
    """

    numerator = np.squeeze(
        np.nansum(
            a.data,
            axis=a.dims.index('hour_of_day')
        )
    )

    denominator = np.squeeze(
        np.nansum(
            np.isfinite(a.data),
            axis=a.dims.index('hour_of_day')
        )
    )

    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=(denominator>0)
    ).astype(np.float32)

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
        default=datetime.datetime(1982, 1, 1),
        type=lambda x: datetime.datetime.strptime(x, '%Y%m').date(),
        help='for which year and month (YYYYMM) start computing the means'
    )
    parser.add_argument(
        '--end',
        default=datetime.datetime(2019, 12, 1),
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
    data = {v: monthly_means(ds[v])[None, ...] for v in args.variables}

    # Process the rest of the files, concatenating along the time dimension
    for f in tqdm.tqdm(mm_files[1:], dynamic_ncols=True):
        ds = xr.open_dataset(f)
        time.append(
            ds.time.data[0].astype('datetime64[M]').astype('datetime64[ns]')
        )
        for v in args.variables:
            data[v] = np.concatenate((data[v], monthly_means(ds[v])[None, ...]), axis=0)
    
    # Create an xarray dataset
    ds = xr.Dataset(
        data_vars={
            v: (('time', 'latitude', 'longitude'), data[v])
            for v in args.variables
        },
        coords={
            'time': ('time', time),
            'latitude': ('latitude', ds.latitude.data),
            'longitude': ('longitude', ds.longitude.data)
        }
    )

    # Write to disk
    fname = f"PATMOS-x_v06-monthlymean_{args.start.strftime('%Y%m')}-{args.end.strftime('%Y%m')}.nc"
    ds.to_netcdf(args.destination / fname)