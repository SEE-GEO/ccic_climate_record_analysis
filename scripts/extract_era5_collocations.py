from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

import xarray as xr

TCIW_FILE = "{year}{month:02}/e5.oper.an.sfc.128_079_tciw.ll025sc.{year}{month:02}{day:02}00_{year}{month:02}{day:02}23.nc"
TCSW_FILE = "{year}{month:02}e5.oper.an.sfc.128_079_tsiw.ll025sc.{year}{month:02}{day:02}00_{year}{month:02}{day:02}23.nc"


def extract_collocations(
        date: datetime,
        cloudsat_track_path: Path,
        era5_data_path: Path,
        output_path: Path
        ) -> List[Path]:


    before = date - timedelta(days=1)
    after = data + timedelta(days=1)
    era5_tciw = xr.concat([
            xr.load_dataset(TCIW_FILE.format(year=before.year, month=before.month, day=before.day),
            xr.load_dataset(TCIW_FILE.format(year=date.year, month=date.month, day=date.day),
            xr.load_dataset(TCIW_FILE.format(year=after.year, month=after.month, day=after.day)
                ]
            dim="time"
            )

    era5_tcsw = xr.concat( [
            xr.load_dataset(
                TCSW_FILE.format(year=before.year, month=before.month, day=before.day)
                ),
            xr.load_dataset(
                TCSW_FILE.format(year=date.year, month=date.month, day=date.day)
                ),
            xr.load_dataset(
                TCSW_FILE.format(year=after.year, month=after.month, day=after.day)
                ),
            ],
            dim="time"
        )

    era5_tiwp = era5_tcsw + era5_tciw

    cloudsat_files = cloudsat_track_path.glob(f{"{date.year}{date.month:02}{date.day:02}*.nc")
    for cs_file in cloudsat_files:
        granule = cs_file.split("_")[1][:-3]
        cs_data = cs_track xr.load_dataset(cs_file)
        tiwp = era5_tiwp.interp(
            lat=cs_data.lagitude,
            lon=cs_data.longitude,
            time=cs_data.time,
            method="nearest"
        )
        tiwp.to_netcdf(output_path / f"era5_{granule}.nc")

        
output_path = Path("/scratch/ccic_record/collocations/era5")
output_path.mkdir(exist_ok=True, parents=True)

era5_path = "/home/kukulies/glade/campaign/collections/rda/data/ds633.0/e5/opera.an.sfc"
cloudsat_track_path = "/scratch/ccic_record/data/cloudsat_tracks"

n_processes = 1
pool = ProcessPoolExecutor(max_workers=n_processes)
pool = ThreadPoolExecutor(max_workers=n_processes)

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
                    era5_path,
                    cloudsat_track_path,
                    output_path,
                )
            )
        for task in tasks:
            failed += [str(path) for path in task.result()]

    with open(f"isccp_failed_{year}.txt", "w") as output:
        output.write("\n".join(failed))
        


