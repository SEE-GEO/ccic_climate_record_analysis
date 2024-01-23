"""This script temporarily downloads L3 MODIS datafiles and stores the relevant variables in a netcdf.

When running the script, it must be accompanied by a LAADS DAAC authentication token.
"""

import subprocess
from datetime import datetime, timedelta, timezone
import argparse
from pyhdf.SD import SD, SDC
import netCDF4 as nc
import os
import numpy as np

year = 2017
n_days = 365  # number of days to download (starts at Jan. 1)

# Set up argument parser
parser = argparse.ArgumentParser(description="Download daily data files.")
parser.add_argument("token", help="Authentication token for data download")
args = parser.parse_args()


# def download_data(bash_script_path, source_url, destination_path, token):
#    command = f"bash {bash_script_path} --source {source_url} --destination {destination_path} --token {token}"
#    # Call the Bash script
#    subprocess.run(command, shell=True)


def download_data(bash_script_path, source_url, destination_path, token):
    command = f"bash {bash_script_path} --source {source_url} --destination {destination_path} --token {token}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    downloaded_file_name = result.stdout.strip()  # Assuming the file name is the output
    print(downloaded_file_name)
    # downloaded_file_name = 'modis_data/MOD08_D3.A2015001.061.2017318224316.hdf'
    return downloaded_file_name


def extract_data(date, destination_path, downloaded_file_name, dataset):
    # Construct the file path
    # downloaded_file_name = (
    #    f"MOD08_D3.A{date.strftime('%Y%j')}.061.2017318224316.hdf"  # Modify as needed
    # )
    # file_path = os.path.join(destination_path, downloaded_file_name)
    # with nc.Dataset(nc_filename, 'a') as dataset:
    try:
        daily_data = SD(downloaded_file_name, SDC.READ)

        # print(daily_data.datasets().items().keys)
        cloud_water_path_ice_mean = (
            daily_data.select("Cloud_Water_Path_Ice_Mean").get().astype(np.int32)
        )
        cloud_water_path_ice_histo = (
            daily_data.select("Cloud_Water_Path_Ice_Histogram_Counts")
            .get()
            .astype(np.int32)
        )
        cloud_fraction_ice = (
            daily_data.select("Cloud_Retrieval_Fraction_Ice").get().astype(np.int32)
        )
        cloud_fraction_ice_counts = (
            daily_data.select("Cloud_Retrieval_Fraction_Ice_Pixel_Counts")
            .get()
            .astype(np.int32)
        )
        cloud_fraction = daily_data.select("Cloud_Fraction_Mean").get().astype(np.int32)
        cloud_fraction_counts = (
            daily_data.select("Cloud_Fraction_Pixel_Counts").get().astype(np.int32)
        )
        cloud_fraction_day = (
            daily_data.select("Cloud_Fraction_Day_Mean").get().astype(np.int32)
        )
        cloud_fraction_day_counts = (
            daily_data.select("Cloud_Fraction_Day_Pixel_Counts").get().astype(np.int32)
        )

        date_idx = date.timetuple().tm_yday - 1
        unix_time = int((date - datetime(1970, 1, 1)).total_seconds())
        dataset["date"][date_idx] = unix_time
        dataset["Cloud_Water_Path_Ice_Mean"][date_idx, :, :] = cloud_water_path_ice_mean
        dataset["Cloud_Water_Path_Ice_Histogram_Counts"][
            date_idx, :, :, :
        ] = cloud_water_path_ice_histo
        dataset["Cloud_Retrieval_Fraction_Ice"][date_idx, :, :] = cloud_fraction_ice
        dataset["Cloud_Retrieval_Fraction_Ice_Pixel_Counts"][
            date_idx, :, :
        ] = cloud_fraction_ice_counts
        dataset["Cloud_Fraction_Mean"][date_idx, :, :] = cloud_fraction
        dataset["Cloud_Fraction_Pixel_Counts"][
            date_idx,
            :,
            :,
        ] = cloud_fraction_counts
        dataset["Cloud_Fraction_Day_Mean"][date_idx, :, :] = cloud_fraction_day
        dataset["Cloud_Fraction_Day_Pixel_Counts"][
            date_idx, :, :
        ] = cloud_fraction_day_counts
        dataset["date_idx"][date_idx] = date_idx

    except Exception as e:
        print(f"Error processing {downloaded_file_name}: {e}")


# Example usage
start_date = datetime(year, 1, 1)  # start from Jan 1.
bash_script_path = "./download_modis_daily.sh"
destination_path = "./data_tmp/"

# Initialize NetCDF file
nc_filename = f"ccic_modis_data_{year}.nc"
dataset = nc.Dataset(nc_filename, "w", format="NETCDF4")

dataset.createDimension("date_idx", None)
date_idxs = dataset.createVariable("date_idx", "i4", ("date_idx",))

dataset.createDimension("date", None)
dates = dataset.createVariable("date", "i8", ("date",))
dates.units = "Unix time [s]"

dataset.createDimension("lat", 180)
lats = dataset.createVariable("lat", "f4", ("lat",))
lats[:] = np.arange(-89.5, 90.5, 1)
dataset.createDimension("lon", 360)
lons = dataset.createVariable("lon", "f4", ("lon",))
lons[:] = np.arange(-179.5, 180.5, 1)

dataset.createDimension("histo_bins", 16)
histo_bins = dataset.createVariable(
    "Cloud_Water_Path_Ice_Histo_Intervals", "i2", ("histo_bins",)
)
histo_bins[:] = np.array(
    [5, 15, 35, 75, 125, 175, 225, 275, 325, 375, 425, 475, 750, 1500, 3000, 5000]
)

# Create the variable with dimensions (time, lat, lon)
cloud_water_path_ice_mean = dataset.createVariable(
    "Cloud_Water_Path_Ice_Mean", "i4", ("date", "lat", "lon")
)
cloud_water_path_ice_histo_counts = dataset.createVariable(
    "Cloud_Water_Path_Ice_Histogram_Counts", "i4", ("date", "histo_bins", "lat", "lon")
)
cloud_fraction_ice = dataset.createVariable(
    "Cloud_Retrieval_Fraction_Ice", "i4", ("date", "lat", "lon")
)
cloud_fraction_ice_counts = dataset.createVariable(
    "Cloud_Retrieval_Fraction_Ice_Pixel_Counts", "i4", ("date", "lat", "lon")
)
cloud_fraction = dataset.createVariable(
    "Cloud_Fraction_Mean", "i4", ("date", "lat", "lon")
)
cloud_fraction_counts = dataset.createVariable(
    "Cloud_Fraction_Pixel_Counts", "i4", ("date", "lat", "lon")
)
cloud_fraction_day = dataset.createVariable(
    "Cloud_Fraction_Day_Mean", "i4", ("date", "lat", "lon")
)
cloud_fraction_day_counts = dataset.createVariable(
    "Cloud_Fraction_Day_Pixel_Counts", "i4", ("date", "lat", "lon")
)

current_date = start_date
for i in range(n_days):
    folder_name = current_date.strftime(
        "%j"
    )  # Day of the year as a folder name (e.g., '001')
    source_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD08_D3/{year}/{folder_name}"

    downloaded_file_name = download_data(
        bash_script_path, source_url, destination_path, args.token
    )
    # downloaded_file_name = None
    extract_data(current_date, destination_path, downloaded_file_name, dataset)
    try:
        os.remove(downloaded_file_name)
        print(f"Successfully deleted {downloaded_file_name}")
    except OSError as e:
        print(f"Error deleting {downloaded_file_name}: {e}")
    # Move to the next day
    current_date += timedelta(days=1)

dataset.close()
