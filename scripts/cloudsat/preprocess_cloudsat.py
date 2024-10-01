"""
This script combins DARDAR, 2C-ICE, and 2B-CLDCLASS data and extracts the 
corresponding CCIC GridSat results.
"""

from typing import Any, Dict, List

import numpy as np
import xarray as xr
import os
import glob
from pathlib import Path
import multiprocessing

import data_preprocessing as dp

GRIDSAT_dt = 3  # 3h timesteps for Gridsat


def get_granules(
    year: int, month: int, cloudsat_paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Get available CloudSat granules for a given month.

    Args:
        year: The year
        month: The mont
        cloudsat_path: The paths containing the CloudSat data.

    Return:
        A dictionary mapping granule numbers to corresponding files
        and time stamps.
    """
    granules = {}
    for product in cloudsat_paths:
        files = glob.glob(f"{cloudsat_paths[product]}/{year}/{month:02}/*/*")
        print(product, len(files))
        for f in files:
            file = Path(f)
            filename = file.name
            if filename.endswith(".hdf"):
                tstamp = filename.split("_")[0]
                id_ = filename.split("_")[1]
            elif filename[0:6] == "DARDAR":
                tstamp = filename.split("_")[1]
                id_ = filename.split("_")[2]
            else:
                id_ = 0
                tstamp = 0
                print("No match for ", filename)

            if id_ not in granules.keys():
                granules[id_] = {"files": [file], "tstamp": tstamp}
            else:
                granules[id_]["files"].append(file)
    return granules


def process_granule(
    id_: str, granules: Dict[str, Dict[str, Any]], paths: Dict[str, str]
) -> str:
    """
    Process a granule.

    Args:
        id_: The granule identification string.
        granules: The dictionary containing the files corresponding to each granule.
        paths: A dictionary containing the paths to the output an ccic_datadir.

    Return:
        A message indicating the processing status.
    """
    # no local path needed
    # get basename from the full path
    output_dir = paths["output_dir"]
    ccic_datadir = paths["ccic_datadir"]

    if len(granules[id_]["files"]) == 3:

        tstamp = granules[id_]["tstamp"]
        # Load CloudSat data
        allfiles_ok = 0
        for f in granules[id_]["files"]:
            filename = f.name
            if filename.startswith("DARDAR"):
                try:
                    dardar = dp.load_dardar(f)
                    allfiles_ok += 1
                except Exception:
                    return f"Error loading DARDAR file {f}."
            elif filename.startswith(f"{tstamp}_{id_}_CS_2C-ICE"):
                try:
                    ice2c = dp.load_2cice(f)
                    allfiles_ok += 1
                except Exception:
                    return f"Error loading 2C-CIE file {f}."
            elif filename.startswith(f"{tstamp}_{id_}_CS_2B-CLDCLASS"):
                try:
                    cldclass = dp.load_cldclass(f)
                    allfiles_ok += 1
                except Exception:
                    return f"Error loading 2B-CLDCLASS file {f}."
            else:
                print(f"File not recognized: {filename}")
        if allfiles_ok == 3:
            # Process CloudSat data
            dardar = dp.process_dardar(dardar)
            ice2c = dp.process_2cice(ice2c, cldclass)

            # get first and last times for CloudSat granule as a datetime object
            start, stop = dp.get_time_range(dardar["time"])
            # Generate local ccic time grid
            ccic_t_grid = dp.get_local_time_grid(start, stop, GRIDSAT_dt)

            # Load CCIC data
            try:
                ccic_ds = dp.load_ccic_grid(ccic_t_grid, ccic_datadir)
            except Exception as e:
                ccic_ds = None
                print(f"ERROR loading ccic data: ", e)

            # get CloudSat coordinates
            times = dardar["time"]
            lats = dardar["lat"]
            lons = dardar["lon"]
            coords = xr.Dataset(
                {
                    "time": (("rays"), times),
                    "lons": (("rays"), lons),
                    "lats": (("rays"), lats),
                }
            )
            if ccic_ds is not None:
                # Interpolate CCIC data
                result = ccic_ds.interp(
                    time=coords.time, latitude=coords.lats, longitude=coords.lons
                )
            else:
                # set values to nan
                result = xr.Dataset(
                    {
                        "tiwp": (("rays",), np.empty(lats.shape) * np.nan),
                        "cloud_prob_2d": (("rays",), np.empty(lats.shape) * np.nan),
                    }
                )
            # Create dataset to save
            ds = xr.Dataset(
                data_vars=dict(
                    iwp_ccic=(["rays"], result["tiwp"].data),
                    iwp_2cice=(["rays"], ice2c["iwp"]),
                    iwp_dardar=(["rays"], dardar["iwp"]),
                    cloud_mask_ccic=(["rays"], result["cloud_prob_2d"].data),
                    cloud_mask_2cice=(["rays"], ice2c["cm"]),
                    cloud_mask_dardar=(["rays"], dardar["cm"]),
                ),
                coords=dict(
                    rays=np.arange(lons.size),
                    longitude=(["rays"], lons),
                    latitude=(["rays"], lats),
                    time=(["rays"], times),
                ),
                attrs=dict(
                    cloudsat_granule=id_,
                    cloudsat_timestamp=tstamp,
                ),
            )
            # Name of output file
            outfile = f"ccicgridsat_dardar_2cice_{tstamp}_{id_}.nc"
            outpath = os.path.join(output_dir, outfile)
            ds.to_netcdf(outpath)
            message = f"Processesing done for granule {id_}"
        else:
            message = f"Maybe wierd filepaths for granule {id_}"
    else:
        message = f"Incomplete data for granule {id_}"
    return message


def run(args: Dict[str, Any]) -> str:
    """
    Wrapper function to process a single granule.

    Args:
        args: A dictionary containing the arguments to the process_granule function.

    Return:
        A string indicating the processing statue.
    """
    id_ = args["id_"]
    granules = args["granules"]
    paths = args["paths"]
    return process_granule(id_, granules, paths)


if __name__ == "__main__":

    # Path to CCIC data
    datadir_ccic = "/home/spfrnd/sun/ccic/gridsat/"
    # Paths to CloudSat data
    datadir_dardar = "/data/s6/L2_ext/DARDAR/dardar3.1/"
    datadir_2cice = "/data/s6/L2_ext/2C-ICE/"
    datadir_cldclass = "/data/s6/L2_ext/2B-CLDCLASS/"

    # path to store output data
    output_dir = "/scratch/ccic_climate_record/data/cloudsat"
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "output_dir": output_dir,
        "ccic_datadir": datadir_ccic,
    }

    cloudsat_paths = {
        "dardar": datadir_dardar,
        "2cice": datadir_2cice,
        "cldclass": datadir_cldclass,
    }

    years = np.arange(2012, 2020)
    for yr in years:

        for month in range(1, 12):

            # Get dict with existing granules
            granules = get_granules(yr, month, cloudsat_paths)
            arguments = [
                {"id_": id_, "granules": granules, "paths": paths} for id_ in granules
            ]

            # the number of CPU cores to use
            nprocesses = 32  # multiprocessing.cpu_count()
            count = 1
            with multiprocessing.Pool(processes=nprocesses) as pool:
                for id_message in pool.imap(run, arguments):
                    print(f"{count}/{len(arguments)} {id_message}")
                    count += 1

            print(f"Finished processing {yr}, month {str(month).zfill(2)}")
