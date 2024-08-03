#!/usr/bin/env python3
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
from pathlib import Path
from filelock import FileLock

import numpy as np
import xarray as xr

from pansat import Product, TimeRange
from pansat.time import to_datetime
from pansat.products.satellite.cloudsat import l2b_cldclass_lidar
from pansat.products.satellite.ncei import isccp_hgg


LOGGER = logging.getLogger(__name__)


def extract_collocations(
        date: datetime,
        output_path: Path,
        ref_product: Product,
        delta_t: np.timedelta64 = np.timedelta64(15, "m")
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
    LOGGER.info("Processing day %s", date)
    output_path = Path(output_path)
    date = to_datetime(date)
    start = datetime(date.year, date.month, date.day)
    end = start + timedelta(hours=23, minutes=59, seconds=59)

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
    failed = []

    for cs_rec in cs_recs:
        cs_rec = cs_rec.get()

        try:
            # Retrieve file from local machine or remote
            cs_data = ref_product.open(cs_rec)

            tr = cs_rec.temporal_coverage
            lock = FileLock("isccp.lock")
            with lock:
                isccp_recs = isccp_hgg.find_files(tr)
                isccp_data = xr.concat(
                    [xr.load_dataset(rec.get().local_path) for rec in isccp_recs],
                    "time"
                )
            isccp_data = isccp_data[[
                "cldamt_types",
                "wp_type",
                "scene"
            ]]
            isccp_data.coords['lon'] = (isccp_data.coords['lon'] + 180) % 360 - 180
            isccp_data = isccp_data.sortby(isccp_data.lon)
            isccp_data = isccp_data.interp(
                lat=cs_data.latitude,
                lon=cs_data.longitude,
                time=cs_data.time,
                method="nearest"
            )

            start_time = to_datetime(cs_data.time[0].data)
            start_time_str = start_time.strftime("%Y%m%d%H%M%S")

            granule = cs_rec.filename.split("_")[1]
            output_filename = f"isccp_{start_time_str}_{granule}.nc"

            isccp_data.to_netcdf(output_path / output_filename)
        except:
            LOGGER.exception(
                "Encountered an error while processing CloudSat file %s.",
                cs_rec.filename
            )
            failed.append(cs_rec.local_path)

    return failed



logging.basicConfig(level="INFO", force=True)

output_path = Path("/scratch/ccic_record/collocations/isccp")
output_path.mkdir(exist_ok=True, parents=True)

n_processes = 16
pool = ProcessPoolExecutor(max_workers=n_processes)
#pool = ThreadPoolExecutor(max_workers=n_processes)

year_start = 2007
year_end = 2020

for year in range(year_start, year_end):

    tasks = []
    failed = []

    for month in range(0, 12):
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

    with open(f"isccp_failed_{year}.txt", "w") as output:
        output.write("\n".join(failed))
