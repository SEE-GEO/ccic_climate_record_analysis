"""
This notebook computes cloud amount and TIWP time series
from the precomputed CCIC CPCIR monthly means record.

The core code is based on
../notebooks/ccic_gridsat_time_series.ipynb
"""
import logging
from pathlib import Path

import numpy as np
import tqdm
import xarray as xr

logging.basicConfig(level=logging.INFO)

files = sorted(list(Path('/scratch/ccic_record/data/ccic/cpcir/monthly_means').glob('ccic_cpcir_*_monthlymean.zarr')))

logging.info(f"Found {len(files)} files.")

# Load the mask
logging.info("Loading the mask")
mask = xr.load_dataset("/scratch/ccic_record/data/mask_24.nc").mask

logging.info("Interpolating mask to CPCIR coordinates")
ds = xr.open_zarr(files[0]) # Get coordinates from one file
mask = mask.astype(int).interp(
    {
        'latitude': ds.latitude.data,
        'longitude': ds.longitude.data
    },
    method='nearest'
) == 1 # .astype(bool) doesn't work well, since NaNs are converted to True

# The code below iterates over all monthly CCIC CPCIR files and
# calculates mean field of cloud amount (``ca``) and
# total ice water path (``tiwp``)
# as well as time series of their area-weighted means and masked means
tiwp_sum = None
tiwp_cnt = None
tiwp_mean = None
tiwp_mean_masked = None
ca_sum = None
ca_cnt = None
ca_mean = None
ca_mean_masked = None
valid_frac = None
valid_frac_masked = None
time = []

files = files[:2]

for path in tqdm.tqdm(files, dynamic_ncols=True):
    with xr.open_dataset(path, engine='zarr') as input_data:

        tiwp = input_data.tiwp.data[0]
        ca = input_data.cloud_prob_2d.data[0]
        time.append(input_data.month.data[0])
        tiwp_0 = np.nan_to_num(tiwp, nan=0.0, copy=True)
        ca_0 = np.nan_to_num(ca, nan=0.0, copy=True)

        weights = np.cos(np.deg2rad(input_data.latitude.data[..., None]))
        weights = np.broadcast_to(weights, tiwp.shape)
        weights_sum = weights.sum()
        weights_masked = weights * mask
        weights_masked_sum = weights_masked.sum()

        if tiwp_sum is None:
            tiwp_sum = tiwp_0
            tiwp_cnt = np.isfinite(tiwp).astype("float32")
            ca_sum = ca_0
            ca_cnt = np.isfinite(ca).astype("float32")
            tiwp_mean = [(tiwp_0 * weights).sum() / (weights * np.isfinite(tiwp)).sum()]
            tiwp_mean_masked = [
                (tiwp_0 * weights_masked).sum()
                / (weights_masked * np.isfinite(tiwp)).sum()
            ]
            ca_mean = [(ca_0 * weights).sum() / (weights * np.isfinite(ca)).sum()]
            ca_mean_masked = [
                (ca_0 * weights * mask).sum() / (weights_masked * np.isfinite(ca)).sum()
            ]
            valid_frac = [(weights * np.isfinite(ca)).sum() / weights_sum]
            valid_frac_masked = [(weights * np.isfinite(ca)).sum() / weights_masked_sum]
        else:
            tiwp_sum += tiwp_0
            tiwp_cnt += np.isfinite(tiwp).astype("float32")
            ca_sum += ca_0
            ca_cnt += np.isfinite(ca).astype("float32")
            tiwp_mean += [
                (tiwp_0 * weights).sum() / (weights * np.isfinite(tiwp)).sum()
            ]
            tiwp_mean_masked += [
                (tiwp_0 * weights_masked).sum()
                / (weights_masked * np.isfinite(tiwp)).sum()
            ]
            ca_mean += [(ca_0 * weights).sum() / (weights * np.isfinite(ca)).sum()]
            ca_mean_masked += [
                (ca_0 * weights * mask).sum() / (weights_masked * np.isfinite(ca)).sum()
            ]
            valid_frac += [(weights * np.isfinite(ca)).sum() / weights_sum]
            valid_frac_masked += [
                (weights * np.isfinite(ca)).sum() / weights_masked_sum
            ]

        latitude = input_data.latitude.data
        longitude = input_data.longitude.data

time = np.array(time)

results = xr.Dataset(
    {
        "tiwp": (
            ("latitude", "longitude"),
            np.divide(
                tiwp_sum,
                tiwp_cnt,
                out=np.full_like(tiwp_sum, np.nan),
                where=tiwp_cnt > 0
            )
        ),
        "ca": (
            ("latitude", "longitude"),
            np.divide(
                ca_sum,
                ca_cnt,
                out=np.full_like(ca_sum, np.nan),
                where=ca_cnt > 0
            )
        ),
        "tiwp_mean": (("time",), np.array(tiwp_mean)),
        "tiwp_mean_masked": (("time",), np.array(tiwp_mean_masked)),
        "ca_mean": (("time",), np.array(ca_mean)),
        "ca_mean_masked": (("time",), np.array(ca_mean_masked)),
        "valid_frac": (("time",), np.array(valid_frac)),
        "valid_frac_masked": (("time",), np.array(valid_frac_masked)),
    },
    {
        "time": ("time", time),
        "latitude": ("latitude", latitude),
        "longitude": ("longitude", longitude)
    }
)

results.to_netcdf('/scratch/ccic_record/data/ccic/cpcir/ccic_cpcir_1998-2023_monthlymeans.nc')