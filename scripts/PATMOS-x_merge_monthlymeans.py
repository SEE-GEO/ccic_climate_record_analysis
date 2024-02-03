"""
Merge the monthly means files into a single netCDF file
"""

import argparse
import datetime
from pathlib import Path

from dask.diagnostics import ProgressBar
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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source',
    required=True,
    type=Path,
    help=(
        "path to the location of the netCDFs "
        "generated with PATMOS-x_monthlymean.py"
    )
)
parser.add_argument(
    '--destination',
    required=True,
    type=Path,
    help="path to the directory where to save the new netCDFs"
)
parser.add_argument(
    '--start',
    default=datetime.datetime(1981, 8, 1).date(),
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
    list(args.source.glob('PATMOS-x_v06-monthlymean_*.nc'))
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

# Open all files
ds = xr.open_mfdataset(mm_files)
with ProgressBar():
    print(f"Loading {len(mm_files)} files")
    ds.load()

# Write to disk
fname = f"PATMOS-x_v06-monthlymean_{args.start.strftime('%Y%m')}-{args.end.strftime('%Y%m')}.nc"
fpath = args.destination / fname
delayed_job = ds.to_netcdf(
    fpath,
    encoding={var: {'zlib': True, 'complevel': 9} for var in ds},
    compute=False
)
with ProgressBar():
    print(f"Saving {fpath}")
    delayed_job.compute()