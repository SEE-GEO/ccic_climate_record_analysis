import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as datetime
import pandas as pd
from scipy.interpolate import griddata
from multiprocessing import Pool

mask_filepath = "/scratch/ccic_record/data/mask_24.nc"


def calculate_statistics(dataset):
    """
    Manually calculate annual statistics for zonal mean plots, etc.
    This is not used, but it is useful for figuring out statistics and checking results.
    """

    # dataset = xr.open_dataset(file_path)

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
        "TIWP_global_mean": calculate_global_mean(IWP * CF_ice, latitude, longitude),
        "CF_global_mean": calculate_global_mean(CF, latitude, longitude),
    }

    return statistics


def calculate_global_mean(data, dataset):
    """
    Calculate a global mean, weighted by grid area at a given latitude
    Assumes data is 2D array, gridded on to the lats and lons provided as arguments.
    """
    factor = np.cos(np.deg2rad(dataset.lat))
    factor.name = "factor"
    global_mean = data.weighted(factor).mean(("lon", "lat", "date"))
    return global_mean


def calculate_monthly_means(dataset, variable, mask):
    """
    Calculate monthly global means from a yearly dataset.
    If mask is True, then interpolate and mask the data (though this is
    not needed is using the '_masked' datasets.
    """

    processed_results = {}
    if dataset["date"].dtype.kind != "M":
        valid_dates = dataset.date.where(dataset["date"] >= 0)
        dataset["date"] = (
            "date",
            pd.to_datetime(valid_dates.values, unit="s", errors="coerce"),
        )
    monthly_data = dataset.groupby("date.month")

    for month, group in monthly_data:
        if variable == "TIWP":
            IWP = group.Cloud_Water_Path_Ice_Mean
            CF_ice = group.Cloud_Retrieval_Fraction_Ice
            data_filtered = IWP.where(IWP >= 0) * CF_ice.where(CF_ice >= 0) * 0.0001
        elif variable == "CF":
            CF = group.Cloud_Fraction_Mean
            data_filtered = CF.where(CF >= 0) * 0.0001

        processed_results[month] = calculate_global_mean(data_filtered, dataset)

    return processed_results


def process_monthly_means(directory_path, mask=False):
    """
    Calculates monthly means for all data in a directory.
    Assumes this is annual modis data.
    Output is a dataframe containing date (year and month) and the means.
    """
    all_results = []

    file_suffix = "_masked.nc" if mask else ".nc"

    for file in os.listdir(directory_path):
        if file.endswith(file_suffix):
            file_path = os.path.join(directory_path, file)
            if "_masked.nc" in file_path:
                suffix_for_variable = "masked"
            else:
                suffix_for_variable = "unmasked"
            dataset = xr.open_dataset(file_path)
            latitude = dataset["lat"][:]
            longitude = dataset["lon"][:]

            dates_unix = dataset["date"][:]
            dates_unix = np.where(
                dates_unix == -9223372036854775806, 0, dates_unix
            )  # this value indicates a failed download of l3 daily

            year = datetime.datetime.fromtimestamp(dates_unix[-1]).year

            cf_monthly_mean = calculate_monthly_means(dataset, "CF", mask=mask)
            tiwp_monthly_mean = calculate_monthly_means(dataset, "TIWP", mask=mask)

            cf_variable_name = f"CF_global_mean_{suffix_for_variable}"
            tiwp_variable_name = f"TIWP_global_mean_{suffix_for_variable}"

            for month in cf_monthly_mean.keys():
                date = pd.to_datetime(f"{year}-{int(month):02d}-01")
                all_results.append(
                    {
                        "date": date,
                        tiwp_variable_name: tiwp_monthly_mean[month].values.item(),
                        cf_variable_name: cf_monthly_mean[month].values.item(),
                    }
                )

    all_results_df = pd.DataFrame(all_results)

    return all_results_df


def interpolate_and_mask(args):
    points_original, data, points_target, mask_array, lon_grid_target_shape = args
    data_interpolated = griddata(
        points_original,
        data.flatten(),
        points_target,
        method="nearest",
        fill_value=np.nan,
    ).reshape(lon_grid_target_shape)

    data_masked = np.where(mask_array, data_interpolated, np.nan)

    return np.ma.masked_invalid(data_masked)


def mask_data(data, lat_original, lon_original):

    mask_ds = xr.open_dataset(mask_filepath)
    mask_array = mask_ds.mask.values

    lon_grid_original, lat_grid_original = np.meshgrid(lon_original, lat_original)
    points_original = np.array(
        [lon_grid_original.flatten(), lat_grid_original.flatten()]
    ).T

    lon_target = mask_ds.longitude.values
    lat_target = mask_ds.latitude.values[::-1]  # reverse to match modis l3
    lon_grid_target, lat_grid_target = np.meshgrid(lon_target, lat_target)
    points_target = np.array([lon_grid_target.flatten(), lat_grid_target.flatten()]).T
    lon_grid_target_shape = lon_grid_target.shape

    with Pool() as pool:
        args = [
            (points_original, data[i], points_target, mask_array, lon_grid_target_shape)
            for i in range(data.shape[0])
        ]
        print(points_original.shape)
        print(data[0].shape)
        data_new = pool.map(interpolate_and_mask, args)

    return np.array(data_new), lon_target, lat_target


def create_masked_dataset(year):
    """
    Interpolate modis data onto gridsat lat/lons and then mask.
    Ran as a pre-processing step to create the '_masked' files.
    """

    nc_filename_original = f"/scratch/ccic_record/data/modis/ccic_modis_data_{year}.nc"
    nc_filename_masked = (
        f"/scratch/ccic_record/data/modis/ccic_modis_data_{year}_masked.nc"
    )

    # Load original dataset
    original_ds = xr.open_dataset(nc_filename_original)
    dataset = nc.Dataset(nc_filename_masked, "w", format="NETCDF4")

    lat_original, lon_original = original_ds.lat.values, original_ds.lon.values

    mask_ds = xr.open_dataset(mask_filepath)

    dataset.createDimension("date_idx", None)
    date_idxs = dataset.createVariable("date_idx", "i4", ("date_idx",))

    dataset.createDimension("date", None)
    dates = dataset.createVariable("date", "i8", ("date",))
    dates[:] = original_ds.date.values
    dates.units = "Unix time [s]"

    dataset.createDimension("lat", len(mask_ds.latitude.values[::-1]))
    lats = dataset.createVariable("lat", "f4", ("lat",))
    lats[:] = mask_ds.latitude.values[::-1]  # reverse to match modis L3
    dataset.createDimension("lon", len(mask_ds.longitude.values[::-1]))
    lons = dataset.createVariable("lon", "f4", ("lon",))
    lons[:] = mask_ds.longitude.values

    cloud_water_path_ice_mean = dataset.createVariable(
        "Cloud_Water_Path_Ice_Mean", "i4", ("date", "lat", "lon")
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

    for var_name in original_ds.data_vars:
        if var_name not in dataset.variables:
            continue  # skip counts, not needed
        original_data = original_ds[var_name].values
        # mask and interpolation
        print(var_name)
        processed_data, _, _ = mask_data(original_data, lat_original, lon_original)
        dataset[var_name][:, :, :] = processed_data

    # Close the datasets
    original_ds.close()
    dataset.close()


if __name__ == "__main__":

    directory_path = "/scratch/ccic_record/data/modis/"

    df_unmasked = process_monthly_means(directory_path, mask=False)
    df_unmasked.set_index("date", inplace=True)
    ds = df_unmasked.to_xarray().sortby("date")
    ds.to_netcdf("modis_cf_tiwp_time_series.nc")
