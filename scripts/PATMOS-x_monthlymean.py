"""
Compute pixel-wise monthly means separated by hour of day
"""

import argparse
from pathlib import Path

import numpy as np
import tqdm
import xarray as xr

def get_files_by_month(source: Path) -> dict:
    """Get files by month

    Args:
        source: path to the folder containing the PATMOS-x netCDFs

    Returns:
        A dictionary with {YYYYMM: [file1, file2, ..]}
    """
    files = list(source.glob('*nc'))
    dictionary = dict()
    for f in files:
        month = f.name.split('_')[4][1:-2]
        if month not in dictionary:
            dictionary[month] = []
        dictionary[month] = dictionary[month] + [f]
    return dictionary

def process_month(files: list[Path]) -> xr.Dataset:
    """
    Compute the monthly mean for one month

    Args:
        files: list with Paths to the files in the given month

    Returns:
        A dataset with the computed monthly means
    """

    # Create an empty dataset to later populate
    ds = xr.load_dataset(files[0])
    ds = ds.drop_vars(ds.keys())
    shape = (24, 1, ds.latitude.size, ds.longitude.size)
    ds['hour_of_day'] = (('hour_of_day',), range(24))
    ds['cloud_probability_count'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=int))
    ds['cloud_probability_sum'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=float))
    ds['cloud_fraction_count'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=int))
    ds['cloud_fraction_sum'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=float))
    ds['tiwp_count'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=int))
    ds['tiwp_sum'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=float))
    ds['tiwp_mixed_count'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=int))
    ds['tiwp_mixed_sum'] = (('hour_of_day', 'time', 'latitude', 'longitude'), np.zeros(shape, dtype=float))
    ds.attrs = {'source': [str(f.name) for f in files]}

    for f in tqdm.tqdm(files, dynamic_ncols=True, leave=False):
        ds_f = xr.open_dataset(f)
        hour_index = (ds_f.scan_line_time.data // 3600).astype(int)
        for i in range(24):
            ds['cloud_probability_sum'][i].data[hour_index == i] += ds_f['cloud_probability'].data[hour_index == i]
            ds['cloud_probability_count'][i].data[hour_index == i] += np.ones((hour_index == i).sum(), dtype=int)
            ds['cloud_fraction_sum'][i].data[hour_index == i] += ds_f['cloud_fraction'].data[hour_index == i]
            ds['cloud_fraction_count'][i].data[hour_index == i] += np.ones((hour_index == i).sum(), dtype=int)
            tiwp = np.where(ds_f['cloud_phase'] == 4, ds_f['cld_cwp_dcomp'], np.nan)
            ds['tiwp_sum'][i].data[hour_index == i] += np.where(np.isfinite(tiwp.data), tiwp.data, 0)[hour_index == i]
            ds['tiwp_count'][i].data[hour_index == i] += np.where(np.isfinite(tiwp.data), 1, 0)[hour_index == i]
            tiwp_mixed = np.where(np.isin(ds_f['cloud_phase'], [3, 4]), ds_f['cld_cwp_dcomp'], np.nan)
            ds['tiwp_mixed_sum'][i].data[hour_index == i] += np.where(np.isfinite(tiwp_mixed.data), tiwp_mixed.data, 0)[hour_index == i]
            ds['tiwp_mixed_count'][i].data[hour_index == i] += np.where(np.isfinite(tiwp_mixed.data), 1, 0)[hour_index == i]

    ds['cloud_probability'] = (
        ds.cloud_probability_sum.dims,
        np.divide(
            ds['cloud_probability_sum'].data,
            ds['cloud_probability_count'].data,
            out=np.full_like(ds['cloud_probability_sum'].data, np.nan),
            where=ds['cloud_probability_count'].data > 0
        ).astype(np.float32)
    )

    ds['cloud_fraction'] = (
        ds.cloud_probability_sum.dims,
        np.divide(
            ds['cloud_fraction_sum'].data,
            ds['cloud_fraction_count'].data,
            out=np.full_like(ds['cloud_fraction_sum'].data, np.nan),
            where=ds['cloud_fraction_count'].data > 0
        ).astype(np.float32)
    )

    ds['tiwp'] = (
        ds.cloud_probability_sum.dims,
        np.divide(
            ds['tiwp_sum'].data,
            ds['tiwp_count'].data,
            out=np.full_like(ds['tiwp_sum'].data, np.nan),
            where=ds['tiwp_count'].data > 0
        ).astype(np.float32)
    )

    ds['tiwp_mixed'] = (
        ds.cloud_probability_sum.dims,
        np.divide(
            ds['tiwp_mixed_sum'].data,
            ds['tiwp_mixed_count'].data,
            out=np.full_like(ds['tiwp_mixed_sum'].data, np.nan),
            where=ds['tiwp_mixed_count'].data > 0
        ).astype(np.float32)
    )
    
    ds = ds.drop_vars(['cloud_probability_sum', 'cloud_fraction_sum', 'tiwp_sum', 'tiwp_mixed_sum'])

    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="folder containing downloaded PATMOS-x data"
    )
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="folder to place the computed monthly means"
    )

    args = parser.parse_args()

    # Get files
    files = get_files_by_month(args.source)

    for month in tqdm.tqdm(sorted(list(files.keys())), dynamic_ncols=True):
        ds = process_month(files[month])
        ds.to_netcdf(args.destination / f'PATMOS-x_v06-monthlymean_{month}.nc')