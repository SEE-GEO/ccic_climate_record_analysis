#!/usr/bin/env python3
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List

import xarray as xr

from pansat import Product, TimeRange
from pansat.time import to_datetime
from pansat.products.satellite.cloudsat import mod06_1km_aux


LOGGER = logging.getLogger(__name__)


def extract_collocations(
        date: datetime,
        output_path: Path,
) -> List[str]:
    """
    Extract MODIS collocated water path and cloud phase.

    Args:
        date: A datetime object specifying the day for which to extract
            the collocation.
        output_path: The directory to which to write the extracted
            collocations.
        ref_product: The CloudSat reference product from which to extract
            latitude and longitude coordinates.
    """
    output_path = Path(output_path)
    date = to_datetime(date)
    start = datetime(date.year, date.month, date.day)
    end = start + timedelta(hours=23, minutes=59, seconds=59)
    cs_recs = mod06_1km_aux.find_files(TimeRange(start, end))

    failed = []
    for cs_rec in cs_recs:

        # Retrieve file from local machine or remote
        try:
            cs_rec = cs_rec.get()
            cs_data = mod06_1km_aux.open(cs_rec)
            granule = cs_rec.filename.split("_")[1]
            start_time = to_datetime(cs_data.time[0].data)
            start_time_str = start_time.strftime("%Y%m%d%H%M%S")
            output_filename = f"modis_{start_time_str}_{granule}.nc"
            cs_data.to_netcdf(output_path / output_filename)

        except Exception:
            LOGGER.exception(
                "An error was encountered when processing file '%s'.",
                cs_rec.filename
            )
            failed.append(cs_rec.filename)
            continue
    return failed

output_path = Path("/data/ccic/collocations/modis")
output_path.mkdir(exist_ok=True)

n_processes = 4
year_start = 2017
year_end = 2020

for year in range(year_start, year_end):

    failed = []

    for month in range(0, 12):
        tasks = []
        _, n_days = monthrange(year, month + 1)
        pool = ProcessPoolExecutor(max_workers=n_processes)
        for day in range(n_days):
            date = datetime(year, month + 1, day + 1)
            tasks.append(pool.submit(
                extract_collocations,
                date,
                output_path
            ))

        for task in tasks:
            failed += task.result()

    with open(f"modis_failed_{year}.txt", "w") as output:
        output.write("\n".join(failed))
