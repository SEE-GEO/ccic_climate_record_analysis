from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import xarray as xr

TCIW_FILE = "{year}{month:02}/e5.oper.an.sfc.128_079_tciw.ll025sc.{year}{month:02}0100_{year}{month:02}{n_days}23.nc"
TCSW_FILE = "{year}{month:02}/e5.oper.an.sfc.228_090_tcsw.ll025sc.{year}{month:02}0100_{year}{month:02}{n_days:02}23.nc"


def extract_collocations(
    date: datetime, cloudsat_track_path: Path, era5_data_path: Path, output_path: Path) -> List[Path]:

    era5_data_path = Path(era5_data_path)
    cloudsat_track_path = Path(cloudsat_track_path)

    _, n_days = monthrange(date.year, date.month)
    before = date - timedelta(days=n_days)
    _, n_days_before = monthrange(before.year, before.month)
    after = date + timedelta(days=n_days)
    _, n_days_after = monthrange(after.year, after.month)
    era5_tciw = xr.concat(
        [
            # xr.load_dataset(era5_data_path / TCIW_FILE.format(year=before.year, month=before.month, n_days=n_days_before)),
            xr.load_dataset(
                era5_data_path
                / TCIW_FILE.format(year=date.year, month=date.month, n_days=n_days)
            ),
            # xr.load_dataset(era5_data_path / TCIW_FILE.format(year=after.year, month=after.month, n_days=n_days_after))
        ],
        dim="time",
    )

    era5_tcsw = xr.concat(
        [
            # xr.load_dataset(
            #    era5_data_path /
            #    TCSW_FILE.format(year=before.year, month=before.month, n_days=n_days_before)
            #    ),
            xr.load_dataset(
                era5_data_path
                / TCSW_FILE.format(year=date.year, month=date.month, n_days=n_days)
            ),
            # xr.load_dataset(
            #    era5_data_path /
            #    TCSW_FILE.format(year=after.year, month=after.month, n_days=n_days_after)
            #    ),
        ],
        dim="time",
    )

    era5_tiwp = era5_tcsw.TCSW + era5_tciw.TCIW
    lons = era5_tiwp.longitude.data
    lons[lons > 180] -= 360
    era5_tiwp = era5_tiwp.sortby("longitude")

    failed = []

    cloudsat_files = []
    for day in range(n_days):
        jday = (date + timedelta(days=day)).timetuple().tm_yday
        cloudsat_files += cloudsat_track_path.glob(f"{date.year}{jday:03}*.nc")

    for cs_file in cloudsat_files:
        try:
            cs_data = xr.load_dataset(cs_file)

            tiwp = era5_tiwp.interp(
                latitude=cs_data.latitude,
                longitude=cs_data.longitude,
                time=cs_data.time,
                method="nearest",
            )

            granule = cs_file.name.split("_")[1][:-3]
            start_time = to_datetime(cs_data.time[0].data)
            start_time_str = start_time.strftime("%Y%m%d%H%M%S")
            tiwp.to_netcdf(output_path / f"era5_{start_time_str}_{granule}.nc")
        except Exception:
            failed.append(cs_file)
    return failed


output_path = Path("/data/ccic//collocations/era5")
output_path.mkdir(exist_ok=True, parents=True)

era5_path = "/home/kukulies/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc"
cloudsat_track_path = "/data/ccic/cloudsat_tracks"

n_processes = 1
pool = ProcessPoolExecutor(max_workers=n_processes)

year_start = 2010
year_end = 2010

for year in range(year_start, year_end + 1):

    tasks = []
    failed = []

    for month in range(7, 8):
        date = datetime(year, month + 1, 1)
        tasks.append(
            pool.submit(
                extract_collocations,
                date,
                cloudsat_track_path,
                era5_path,
                output_path,
            )
        )
    for task in tasks:
        failed += [str(path) for path in task.result()]

    with open(f"era5_failed_{year}.txt", "w") as output:
        output.write("\n".join(failed))
