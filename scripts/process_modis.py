import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as datetime
import pandas as pd
from scipy.interpolate import griddata

mask_filepath = "/scratch/ccic_record/data/mask_24.nc"


def calculate_statistics(dataset):
    """
    Manually calculate annual statistics for zonal mean plots, etc.
    This is not used, but it is useful for figuring out statistics and checking results.
    """
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


def process_global_distributions(directory_path, mask):
    """
    Calculate global distributions of TIWP and CF over all years of modis data.
    """

    suffix_for_variable = "masked" if mask else "unmasked"

    lat = np.arange(-89.5, 90.5, 1)
    lon = np.arange(-179.5, 180.5, 1)
    n_lats = 180
    n_lons = 360

    CF_global_distribution_unmasked_count = np.zeros((n_lats, n_lons))
    TIWP_global_distribution_unmasked_count = np.zeros((n_lats, n_lons))
    CF_global_distribution_masked_count = np.zeros((n_lats, n_lons))
    TIWP_global_distribution_masked_count = np.zeros((n_lats, n_lons))

    ds = xr.Dataset(
        {
            "CF_global_distribution_unmasked": (
                ("lat", "lon"),
                np.zeros((n_lats, n_lons)),
            ),
            "TIWP_global_distribution_unmasked": (
                ("lat", "lon"),
                np.zeros((n_lats, n_lons)),
            ),
            "CF_global_distribution_masked": (
                ("lat", "lon"),
                np.zeros((n_lats, n_lons)),
            ),
            "TIWP_global_distribution_masked": (
                ("lat", "lon"),
                np.zeros((n_lats, n_lons)),
            ),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )

    mask = mask.mask == 1

    for file in os.listdir(directory_path):
        if file.endswith(".nc"):
            file_path = os.path.join(directory_path, file)

            for masked in [True, False]:
                suffix_for_variable = "masked" if masked else "unmasked"
                dataset = xr.open_dataset(file_path)
                if masked:
                    for var in dataset.data_vars:
                        dataset[var] = dataset[var].where(mask == 1)

                IWP = dataset.Cloud_Water_Path_Ice_Mean
                IWP = IWP.where(IWP >= 0)

                IWP_counts = dataset.Cloud_Retrieval_Fraction_Ice_Pixel_Counts
                IWP_counts = IWP_counts.where(IWP_counts >= 0)

                CF_ice = dataset.Cloud_Retrieval_Fraction_Ice * 0.0001
                CF_ice = CF_ice.where(CF_ice >= 0)

                CF = dataset.Cloud_Fraction_Mean * 0.0001
                CF = CF.where(CF >= 0)
                CF_counts = dataset.Cloud_Fraction_Pixel_Counts
                CF_counts = CF_counts.where(CF_counts >= 0)

                if masked:
                    ds["CF_global_distribution_masked"] += np.nansum(
                        CF * CF_counts, axis=0
                    )
                    CF_global_distribution_masked_count += np.nansum(CF_counts, axis=0)

                    ds["TIWP_global_distribution_masked"] += np.nansum(
                        IWP * CF_ice * IWP_counts, axis=0
                    )
                    TIWP_global_distribution_masked_count += np.nansum(
                        IWP_counts, axis=0
                    )
                else:
                    ds["CF_global_distribution_unmasked"] += np.nansum(
                        CF * CF_counts, axis=0
                    )
                    CF_global_distribution_unmasked_count += np.nansum(
                        CF_counts, axis=0
                    )

                    ds["TIWP_global_distribution_unmasked"] += np.nansum(
                        IWP * CF_ice * IWP_counts, axis=0
                    )
                    TIWP_global_distribution_unmasked_count += np.nansum(
                        IWP_counts, axis=0
                    )
                dataset.close()

    ds["CF_global_distribution_unmasked"] /= CF_global_distribution_unmasked_count
    ds["TIWP_global_distribution_unmasked"] /= TIWP_global_distribution_unmasked_count
    ds["CF_global_distribution_masked"] /= CF_global_distribution_masked_count
    ds["TIWP_global_distribution_masked"] /= TIWP_global_distribution_masked_count

    return ds


def interpolate_mask(mask_ds, original_ds):
    """
    Interpolate the mask dataset onto modis dataset coordinates
    """
    interpolated_mask = mask_ds.astype(int).interp(
        lat=original_ds.lat, lon=original_ds.lon, method="nearest"
    )
    return interpolated_mask


def apply_mask_to_dataset(original_ds, mask):
    mask = mask.mask == 1
    for var in original_ds.data_vars:
        original_ds[var] = original_ds[var].where(mask == True)

    return original_ds


if __name__ == "__main__":

    directory_path = "/scratch/ccic_record/data/modis/"

    df_unmasked = process_monthly_means(directory_path, mask=False)
    df_unmasked.set_index("date", inplace=True)
    ds = df_unmasked.to_xarray().sortby("date")
    ds.to_netcdf("modis_cf_tiwp_time_series.nc")

    process_global_dist = False

    if process_global_dist:
        mask = xr.open_dataset("../mask_24_for_modis.nc")
        modis_data_dir = "../../data/"

        global_dist = process_global_distributions(modis_data_dir, mask)
        global_dist.to_netcdf("global_distribution_cf_tiwp_modis.nc")
