import xarray as xr
import numpy as np
import datetime as datetime
import pandas as pd
from scipy.interpolate import griddata

mask_filepath = "./ccic_climate_record_analysis/mask_24.nc"


def calculate_statistics(file_path):
    dataset = xr.open_dataset(file_path)

    dates_unix = dataset["date"][:]
    dates_unix = np.where(
        dates_unix == -9223372036854775806, 0, dates_unix
    )  # invalid data
    dates = [
        datetime.datetime.utcfromtimestamp(dates_unix[i]).strftime("%Y-%m-%d")
        for i in range(len(dates_unix))
    ]
    latitude = dataset["lat"][:]
    longitude = dataset["lon"][:]

    IWP = dataset["Cloud_Water_Path_Ice_Mean"][:]
    IWP = np.ma.masked_where((IWP == -9999) | (IWP == -2147483647), IWP)  # invalid data

    IWP_counts = dataset["Cloud_Water_Path_Ice_Histogram_Counts"][:]
    IWP_counts = np.sum(
        np.ma.masked_where(
            (IWP_counts == -9999) | (IWP_counts == -2147483647), IWP_counts
        ),
        axis=1,
    )

    CF_ice = dataset["Cloud_Retrieval_Fraction_Ice"][:]
    CF_ice = np.ma.masked_where((CF_ice == -9999) | (CF_ice == -2147483647), CF_ice)
    CF_ice = CF_ice.astype(float) * 0.0001

    CF_ice_counts = dataset["Cloud_Retrieval_Fraction_Ice_Pixel_Counts"][:]
    CF_ice_counts = np.ma.masked_where(
        (CF_ice_counts == -9999) | (CF_ice_counts == -2147483647), CF_ice_counts
    )

    CF = dataset["Cloud_Fraction_Mean"][:]
    CF = np.ma.masked_where((CF == -9999) | (CF == -2147483647), CF)
    CF = CF.astype(float) * 0.0001

    CF_counts = dataset["Cloud_Fraction_Pixel_Counts"][:]
    CF_counts = np.ma.masked_where(
        (CF_counts == -9999) | (CF_counts == -2147483647), CF_counts
    )

    CF_day = dataset["Cloud_Fraction_Day_Mean"][:]
    CF_day = np.ma.masked_where((CF_day == -9999) | (CF_day == -2147483647), CF_day)
    CF_day = CF_day.astype(float) * 0.0001

    CF_day_counts = dataset["Cloud_Fraction_Day_Pixel_Counts"][:]
    CF_day_counts = np.ma.masked_where(
        (CF_day_counts == -9999) | (CF_day_counts == -2147483647), CF_day_counts
    )

    CF_global_distribution = np.sum(CF * CF_counts, axis=0) / np.sum(CF_counts, axis=0)
    CF_day_global_distribution = np.sum(CF_day * CF_day_counts, axis=0) / np.sum(
        CF_day_counts, axis=0
    )

    TIWP_global_distribution = np.sum(IWP * CF_ice * IWP_counts, axis=0) / np.sum(
        IWP_counts, axis=0
    )

    TIWP_zonal_mean = np.zeros(180)
    TIWP_zonal_count = np.zeros(180)
    CF_zonal_mean = np.zeros(180)
    CF_zonal_count = np.zeros(180)
    for i in range(IWP.shape[0]):
        TIWP_zonal_mean += np.sum(IWP[i] * CF_ice[i], axis=1)
        TIWP_zonal_count += 1 * 360  # +1 for every day and 360 for longitudes
        # equally weighted, so same division even if no data for one of the longitudes
        CF_zonal_mean += np.sum(CF_day[i], axis=1)
        CF_zonal_count += 1 * 360
    TIWP_zonal_mean /= TIWP_zonal_count
    CF_zonal_mean /= CF_zonal_count

    # global means
    factor = np.zeros((len(latitude), len(longitude)))
    for i in range(len(longitude)):
        factor[:, i] = np.cos(np.deg2rad(latitude))
    TIWP_global_mean = np.mean(
        np.sum(IWP * CF_ice * factor, axis=(1, 2)) / np.sum(factor)
    )

    statistics = {
        "CF_global_distribution": CF_global_distribution,
        "CF_day_global_distribution": CF_day_global_distribution,
        "TIWP_global_distribution": TIWP_global_distribution,
        "TIWP_zonal_mean": TIWP_zonal_mean,
        "date": dates,
        "CF_zonal_mean": CF_zonal_mean,
        "TIWP_global_mean": calculate_global_mean(latitude, longitude, IWP * CF_ice),
        "CF_global_mean": calculate_global_mean(latitude, longitude, CF),
    }

    return statistics


def calculate_global_mean(data, latitude, longitude):
    """
    Calculate global mean, weighted by grid area at a given latitude
    Assumes data is 2D array, gridded on to given lats and lons.
    """
    global_mean = 0
    factor = np.zeros((len(latitude), len(longitude)))
    for i in range(len(longitude)):
        factor[:, i] = np.cos(np.deg2rad(latitude))

    for i in range(data.shape[0]):
        if np.all(data[i].mask):
            # if np.all(np.isnan(data[i])):
            continue
        factor_i = np.where(data[i].mask, 0, factor)
        global_mean += np.nansum(data[i] * factor_i, axis=(0, 1)) / np.nansum(factor_i)

    global_mean /= data.shape[0]

    return global_mean


def modis_monthly(dataset, latitude, longitude, variable, mask=True):
    processed_results = {}

    dataset["date"] = dataset["date"].where(dataset["date"] >= 0)
    dataset["date"] = pd.to_datetime(dataset["date"], unit="s")

    monthly_data = dataset.groupby("date.month")

    for month, group in monthly_data:
        data_monthly = group[variable].values
        # data_monthly = np.ma.masked_where((data_monthly == -9999) | (data_monthly == -2147483647), data_monthly)
        data_filtered = np.where(
            (data_monthly == -9999) | (data_monthly == -2147483647),
            np.nan,
            data_monthly,
        )
        if mask == True:
            # mask according to gridsat mask
            data_interpolated, data_masked, lon_target, lat_target = mask_data(
                data_monthly, latitude, longitude
            )
            # data_monthly = np.ma.masked_invalid(data_masked)
            latitude = lat_target
            longitude = lon_target
        # data_monthly = data_monthly.astype(float) * 0.0001

        processed_results[month] = calculate_global_mean(
            data_monthly, latitude, longitude
        )

    return processed_results


def mask_data(data, lat_original, lon_original):
    mask_ds = xr.open_dataset(mask_filepath)

    lon_grid_original, lat_grid_original = np.meshgrid(lon_original, lat_original)
    points_original = np.array(
        [lon_grid_original.flatten(), lat_grid_original.flatten()]
    ).T

    lon_target = mask_ds.longitude.values
    lat_target = mask_ds.latitude.values[::-1]  # reverse to match modis
    lon_grid_target, lat_grid_target = np.meshgrid(lon_target, lat_target)
    points_target = np.array([lon_grid_target.flatten(), lat_grid_target.flatten()]).T

    if data.ndim > 2:
        data_new = np.zeros((data.shape[0], len(lat_target), len(lon_target)))
        for i in range(data.shape[0]):
            data_interpolated = griddata(
                points_original,
                data[i].flatten(),
                points_target,
                method="nearest",
                fill_value=np.nan,
            )
            data_interpolated = data_interpolated.reshape(lon_grid_target.shape)

            mask_array = mask_ds.mask.values
            data_masked = np.where(mask_array, data_interpolated, np.nan)

            data_new[i] = np.ma.masked_invalid(data_masked)
    else:
        data_interpolated = griddata(
            points_original,
            data.flatten(),
            points_target,
            method="nearest",
            fill_value=np.nan,
        )
        data_interpolated = data_interpolated.reshape(lon_grid_target.shape)

        mask_array = mask_ds.mask.values
        data_masked = np.where(mask_array, data_interpolated, np.nan)

        data_new = np.ma.masked_invalid(data_masked)

    return data_interpolated, data_new, lon_target, lat_target
