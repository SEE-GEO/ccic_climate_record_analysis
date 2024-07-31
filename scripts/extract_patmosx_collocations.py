#!/usr/bin/env python3
from calendar import monthrange
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from pansat import Product, TimeRange
from pansat.time import to_datetime, to_datetime64
from pansat.products.satellite.cloudsat import l2b_cldclass_lidar
from pansat.products.satellite.ncei import patmosx


LOGGER = logging.getLogger(__name__)



def extract_collocations(
        date: datetime,
        output_path: Path,
        ref_product: Product,
        delta_t: np.timedelta64 = np.timedelta64(3, "h")
) -> None:
    """
    Extract collocations with CloudSat.

    Args:
        date: A datetime object specifying the day for which to extract
            the collocation.
        output_path: The directory to which to write the extracted
            collocations.
        ref_product: The CloudSat reference product from which to extract
            latitude and longitude coordinates.
        delta_t: The maximum time difference between PATMOS-x retrieval and the collocated
            CloudSat measurements.
    """
    output_path = Path(output_path)
    date = to_datetime(date)
    start = datetime(date.year, date.month, date.day)
    end = start + timedelta(hours=23, minutes=59, seconds=59)

    cs_recs = ref_product.find_files(TimeRange(start, end))
    for cs_rec in cs_recs:

        # Retrieve file from local machine or remote
        cs_rec = cs_rec.get()
        cs_data = ref_product.open(cs_rec)

        # Find PATMOS-x file in time range.
        tr = cs_rec.temporal_coverage
        patmosx_recs = patmosx.find_files(tr)

        vars = [
            "scan_line_time",
            "cloud_fraction",
            "cloud_fraction_uncertainty",
            "cloud_mask",
            "cld_cwp_dcomp",
            "cld_opd_dcomp",
            "dcomp_quality",
            "ice_cloud_probability"
        ]

        cloud_vars_cmb = None

        for p_rec in patmosx_recs:
            p_rec = p_rec.get()
            with xr.open_dataset(p_rec.local_path) as data_p:
                slt = data_p.scan_line_time
                data_p["scan_line_time"] = slt.astype(np.int64)
                cloud_vars = data_p[{"time": 0}][vars].interp(
                    latitude=cs_data.latitude,
                    longitude=cs_data.longitude,
                )
                slt = cloud_vars.scan_line_time.data.astype("timedelta64[ns]")
                start = to_datetime64(patmosx.get_temporal_coverage(p_rec).start)
                cloud_vars["scan_line_time"] = (("rays",), start + slt)

            mask = (
                (cloud_vars.scan_line_time > cs_data.time - 0.5 * delta_t) *
                (cloud_vars.scan_line_time < cs_data.time + 0.5 * delta_t)
            )
            slt = cloud_vars.scan_line_time.data
            LOGGER.info(
                "Found %s collocations in %s.",
                mask.data.sum(),
                p_rec.filename
            )

            for var in vars:
                if np.issubdtype(cloud_vars[var].data.dtype, np.floating):
                    cloud_vars[var].data[~mask] = np.nan
                else:
                    cloud_vars[var].data[~mask] = np.datetime64("NaT")

            if cloud_vars_cmb is None:
                cloud_vars_cmb = cloud_vars
            else:
                age = (cloud_vars["scan_line_time"].data - cs_data.time.data)
                new_age = (cloud_vars_cmb["scan_line_time"].data - sc_data.time.data)
                age_mask = mask * (np.abs(new_age) < np.abs(age))
                for var in vars:
                    cloud_vars_cmb[var].data[age_mask] = cloud_vars[var].data[age_mask]

        start_time = to_datetime(cs_data.time[0].data)
        start_time_str = start_time.strftime("%Y%m%d%H%M%S")

        granule = cs_rec.filename.split("_")[1]
        output_filename = f"patmosx_{start_time_str}_{granule}.nc"

        cloud_vars_cmb.to_netcdf(output_path / output_filename)


logging.basicConfig(level="INFO", force=True)
output_path = Path("/edata1/simon/ccic/patmosx")
output_path.mkdir(exist_ok=True)

for month in range(0, 12):
    _, n_days = monthrange(2015, month + 1)
    for day in range(n_days):
        date = datetime(2015, month + 1, day + 1)
        extract_collocations(date, output_path, l2b_cldclass_lidar)
