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
from pansat.products.satellite.cloudsat import l2b_cldclass_lidar
from pansat.products.satellite.gpm import merged_ir
from pansat.download.providers.cloudsat_dpc import cloudsat_dpc_provider


LOGGER = logging.getLogger(__name__)


def extract_collocations(
        date: datetime,
        output_path: Path,
        ref_product: Product
) -> List[str]:
    """
    Extract collocations with CloudSat.

    Args:
        date: A datetime object specifying the day for which to extract
            the collocation.
        output_path: The directory to which to write the extracted
            collocations.
        ref_product: The CloudSat reference product from which to extract
            latitude and longitude coordinates.

    Return:
        A list of CloudSat reference files for which the processing failed>
    """
    output_path = Path(output_path)
    date = to_datetime(date)
    start = datetime(date.year, date.month, date.day)
    end = start + timedelta(hours=23, minutes=59, seconds=59)

    cs_recs = ref_product.find_files(
            TimeRange(start, end),
            provider=cloudsat_dpc_provider
    )
    failed = []

    for cs_rec in cs_recs:
        try:
            # Retrieve file from local machine or remote
            cs_rec = cs_rec.get()
            cs_data = ref_product.open(cs_rec)

            # Find CPCIR file in time range.
            tr = cs_rec.temporal_coverage
            cpcir_recs = merged_ir.find_files(tr)

            cpcir_data = []
            for cpcir_rec in cpcir_recs:
                cpcir_rec = cpcir_rec.get()
                cpcir_data.append(merged_ir.open(cpcir_rec))
            cpcir_data = xr.concat(cpcir_data, "time")

            cpcir_data = cpcir_data.interp(
                time=cs_data.time,
                latitude=cs_data.latitude,
                longitude=cs_data.longitude
            )

            start_time = to_datetime(cs_data.time[0].data)
            start_time_str = start_time.strftime("%Y%m%d%H%M%S")

            granule = cs_rec.filename.split("_")[1]
            output_filename = f"cpcir_{start_time_str}_{granule}.nc"

            cpcir_data.to_netcdf(output_path / output_filename)

        except Exception:
            LOGGER.exception(
                "An error was encountered when processing CS reference file '%s'.",
                cs_rec.filename
            )
            failed.append(cs_rec.local_path)

    return failed

output_path = Path("/data/ccic/collocations/cpcir")
output_path.mkdir(exist_ok=True)

failed = []
year = 2010

n_processes = 4
pool = ProcessPoolExecutor(max_workers=4)

tasks = []
for month in range(1, 12):
    _, n_days = monthrange(year, month + 1)
    for day in range(n_days):
        date = datetime(year, month + 1, day + 1)
        tasks.append(pool.submit(
            extract_collocations,
            date,
            output_path,
            l2b_cldclass_lidar
        ))

    for task in tasks:
        failed += [str(path) for path in task.result()]

with open(f"cpcir_failed_{year}.txt", "w") as output:
    output.write("\n".join(failed))
