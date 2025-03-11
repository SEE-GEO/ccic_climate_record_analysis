#!/usr/bin/env python3
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from pathlib import Path
from datetime import datetime, timedelta

from filelock import FileLock
import numpy as np
import xarray as xr

from pansat import Product, TimeRange, FileRecord
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

    failed = []

    lock = FileLock("icare.lock")
    with lock:
        cs_recs = ref_product.find_files(TimeRange(start, end))
        for cs_rec in cs_recs:
            cs_rec_local = cs_rec.get()
            try:
                ref_product.open(cs_rec_local)
            except Exception:
                LOGGER.info("Re-downloading %s.", cs_rec.filename)
                cs_rec = cs_rec.download()

    cs_paths = set([rec.get().local_path for rec in cs_recs])
    for cs_path in cs_paths:
        try:
            cs_rec = FileRecord(cs_path, product=ref_product)
            # Retrieve file from local machine or remote
            cs_data = ref_product.open(cs_rec)

            # Find PATMOS-x file in time range.
            tr = cs_rec.temporal_coverage
            patmosx_recs = patmosx.find_files(tr)

            vars = [
                "scan_line_time",
                "cloud_probability",
                "cloud_fraction",
                "cld_iwp_dcomp",
            ]

            cloud_vars_cmb = None

            patmosx_files = []

            for p_rec in patmosx_recs:
                p_rec = p_rec.get()
                with xr.open_dataset(p_rec.local_path, engine="zarr") as data_p:


                    cloud_vars = data_p[{"time": 0}][vars].interp(
                        latitude=cs_data.latitude,
                        longitude=cs_data.longitude,
                    )

                    slt = cloud_vars.scan_line_time.data
                    offset = to_datetime64(p_rec.temporal_coverage.start)
                    invalid = slt < 0
                    second_of_day = slt.astype("int64").astype("timedelta64[s]")
                    slt = offset + second_of_day
                    slt[invalid] = np.datetime64("NAT")
                    slt = slt.astype("datetime64[ns]")
                    cloud_vars["scan_line_time"] = (("rays",), slt)
                    cloud_vars["cs_time"] = (("rays", cs_data.time.data))

                age = cloud_vars.scan_line_time.data - cs_data.time.data
                mask = (
                    (cloud_vars.scan_line_time > cs_data.time - delta_t) *
                    (cloud_vars.scan_line_time < cs_data.time + delta_t) *
                    np.isfinite(cloud_vars["cld_iwp_dcomp"].data)
                )
                slt = cloud_vars.scan_line_time.data

                if mask.data.sum() < 1:
                    continue

                LOGGER.info(
                    "Found %s matching profiles in PATMOS-x file '%s'.",
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
                    new_age = (cloud_vars["scan_line_time"].data - cs_data.time.data)
                    age = (cloud_vars_cmb["scan_line_time"].data - cs_data.time.data)
                    age_mask = mask * ((np.abs(new_age) < np.abs(age)) + np.isnan(age))
                    for var in vars + ["cs_time"]:
                        cloud_vars_cmb[var].data[age_mask] = cloud_vars[var].data[age_mask]
                    if age_mask.any():
                        patmosx_files.append(p_rec.filename)

            if cloud_vars_cmb is None:
                LOGGER.info(
                    "Found no collocations for CloudSat file %s",
                    cs_rec.filename
                )
                continue

            start_time = to_datetime(cs_data.time[0].data)
            start_time_str = start_time.strftime("%Y%m%d%H%M%S")

            granule = cs_rec.filename.split("_")[1]
            output_filename = f"patmosx_{start_time_str}_{granule}.nc"

            LOGGER.info(
                "Writing collocations to '%s'.",
                output_filename
            )
            cloud_vars_cmb.attrs["input_files"] = patmosx_files
            cloud_vars_cmb.to_netcdf(output_path / output_filename)

        except Exception:
            LOGGER.exception(
                "Encountered the following error when processing '%s'.",
                cs_rec.filename
            )
            failed.append(cs_rec.local_path)

    return failed


logging.basicConfig(level="WARNING", force=True)
output_path = Path("/scratch/ccic_record/collocations/patmosx")
output_path.mkdir(exist_ok=True)

n_processes = 32
pool = ProcessPoolExecutor(max_workers=n_processes)

year_start = 2009
year_end = 2020

for year in range(year_start, year_end + 1):

    tasks = []
    failed = []

    for month in range(12):
        _, n_days = monthrange(year, month + 1)
        for day in range(n_days):
            date = datetime(year, month + 1, day + 1)
            tasks.append(
                pool.submit(
                    extract_collocations,
                    date,
                    output_path,
                    l2b_cldclass_lidar
                )
            )
        for task in tasks:
            failed += [str(path) for path in task.result()]

    with open(f"patmos_failed_{year}.txt", "w") as output:
        output.write("\n".join(failed))
