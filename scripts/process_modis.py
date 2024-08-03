import xarray as xr
import numpy as np
import datetime as datetime
import pandas as pd
import os


def calculate_global_mean(data, dataset):
    """
    Calculate a global mean, weighted by grid area at a given latitude
    Parameters:
    - data (np.ndarray): 3D numpy array containing the data values to be averaged,
    - dataset (xarray.Dataset): An xarray Dataset object that must contain the latitude coordinates.
      These are used to calculate the weights for the area-weighted mean. The latitude coordinates
      should be accessible via dataset.lat.

    Returns:
    - float: Global mean value
    """

    factor = np.cos(np.deg2rad(dataset.lat))
    factor.name = "factor"

    # xarray method
    global_mean = data.weighted(factor).mean(("lon", "lat", "date"))

    # manual method
    # global_mean = np.nansum(data * factor, axis=(0, 1, 2)) / (
    #    np.nansum(factor) * data.shape[0] * data.shape[2]
    # )
    return global_mean


def calculate_monthly_means(dataset, variable):
    """
    Calculate monthly global means from a yearly dataset.
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
            data_filtered = (
                IWP.where(IWP >= 0).fillna(0) * CF_ice.where(CF_ice >= 0) * 0.0001
            )
        elif variable == "CF":
            CF = group.Cloud_Fraction_Mean
            data_filtered = CF.where(CF >= 0) * 0.0001

        processed_results[month] = calculate_global_mean(data_filtered, dataset)

    return processed_results


def process_monthly_means(directory_path, mask):
    """
    Calculates monthly means for all data in a directory.
    Assumes this is annual modis data.
    Output is a dataframe containing date (year and month) and the means.
    """
    all_results = []

    mask = mask.mask == 1

    for file in os.listdir(directory_path):
        if file.endswith(".nc"):
            file_path = os.path.join(directory_path, file)

            dataset = xr.open_dataset(file_path)
            dataset_masked = dataset.copy()

            for var in dataset_masked.data_vars:
                dataset_masked[var] = dataset_masked[var].where(mask == 1)

            latitude = dataset["lat"][:]
            longitude = dataset["lon"][:]

            dates_unix = dataset["date"][:]
            dates_unix = np.where(
                dates_unix == -9223372036854775806, 0, dates_unix
            )  # this value indicates a failed download of l3 daily

            year = datetime.datetime.fromtimestamp(dates_unix[-1]).year

            cf_monthly_mean = calculate_monthly_means(dataset, "CF")
            tiwp_monthly_mean = calculate_monthly_means(dataset, "TIWP")
            cf_monthly_mean_masked = calculate_monthly_means(dataset_masked, "CF")
            tiwp_monthly_mean_masked = calculate_monthly_means(dataset_masked, "TIWP")

            for month in cf_monthly_mean.keys():
                date = pd.to_datetime(f"{year}-{int(month):02d}-01")
                all_results.append(
                    {
                        "date": date,
                        "TIWP_global_mean_unmasked": tiwp_monthly_mean[month],
                        "CF_global_mean_unmasked": cf_monthly_mean[month],
                        "TIWP_global_mean_masked": tiwp_monthly_mean_masked[month],
                        "CF_global_mean_masked": cf_monthly_mean_masked[month],
                    }
                )

            dataset.close()
            dataset_masked.close()

    # something wrong with august 2021 modis data - huge amounts are missing for most of the month
    for result in all_results:
        if result["date"] == datetime.datetime(2020, 8, 1, 0, 0, 0):
            result["TIWP_global_mean_unmasked"] = np.nan
            result["CF_global_mean_unmasked"] = np.nan
            result["TIWP_global_mean_masked"] = np.nan
            result["CF_global_mean_masked"] = np.nan

    all_results_df = pd.DataFrame(all_results)

    return all_results_df


def process_global_distributions(directory_path, mask):
    """
    Calculate global distributions of TIWP and CF over all years of modis data.

    Parameters:
    - directory_path (str): Path to the directory containing the modis L3 netcdfs.
    - mask (xarray.DataArray): Mask to apply when calculating the masked distributions.

    Returns:
    - xarray.Dataset: Global distributions of CF and TIWP, both masked and unmasked.
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
                    ds["CF_global_distribution_masked"] += np.nansum(CF, axis=0)
                    CF_global_distribution_masked_count += np.sum(~np.isnan(CF), axis=0)

                    ds["TIWP_global_distribution_masked"] += np.nansum(
                        IWP * CF_ice, axis=0
                    )
                    TIWP_global_distribution_masked_count += np.sum(
                        ~np.isnan(CF_ice), axis=0
                    )

                else:
                    ds["CF_global_distribution_unmasked"] += np.nansum(CF, axis=0)
                    CF_global_distribution_unmasked_count += np.sum(
                        ~np.isnan(CF), axis=0
                    )

                    ds["TIWP_global_distribution_unmasked"] += np.nansum(
                        IWP * CF_ice, axis=0
                    )
                    TIWP_global_distribution_unmasked_count += np.sum(
                        ~np.isnan(CF_ice), axis=0
                    )

                dataset.close()

    ds["CF_global_distribution_unmasked"] /= CF_global_distribution_unmasked_count
    ds["TIWP_global_distribution_unmasked"] /= TIWP_global_distribution_unmasked_count
    ds["CF_global_distribution_masked"] /= CF_global_distribution_masked_count
    ds["TIWP_global_distribution_masked"] /= TIWP_global_distribution_masked_count

    return ds


def process_zonal_means(directory_path, mask):
    """
    Calculate zonal means of TIWP and CF over all years of modis data.
    """

    suffix_for_variable = "masked" if mask else "unmasked"

    lat = np.arange(-89.5, 90.5, 1)
    n_lats = 180

    CF_zonal_mean_unmasked_count = np.zeros((n_lats))
    TIWP_zonal_mean_unmasked_count = np.zeros((n_lats))
    CF_zonal_mean_masked_count = np.zeros((n_lats))
    TIWP_zonal_mean_masked_count = np.zeros((n_lats))

    ds = xr.Dataset(
        {
            "CF_zonal_mean_unmasked": (
                ("lat"),
                np.zeros((n_lats)),
            ),
            "TIWP_zonal_mean_unmasked": (
                ("lat"),
                np.zeros((n_lats)),
            ),
            "CF_zonal_mean_masked": (
                ("lat"),
                np.zeros((n_lats)),
            ),
            "TIWP_zonal_mean_masked": (
                ("lat"),
                np.zeros((n_lats)),
            ),
        },
        coords={
            "lat": lat,
        },
    )

    mask = mask.mask == 1

    for file in os.listdir(directory_path):
        if file.endswith(".nc"):
            file_path = os.path.join(directory_path, file)
            print(file_path)
            if "_masked.nc" in file_path:
                continue
            if "global" in file_path:
                continue
            if "time" in file_path:
                continue
            if "zonal" in file_path:
                continue
            for masked in [True, False]:
                suffix_for_variable = "masked" if masked else "unmasked"
                dataset = xr.open_dataset(file_path)
                if masked:
                    for var in dataset.data_vars:
                        dataset[var] = dataset[var].where(mask == 1)

                # remove failed downloads (negative date)
                dataset = dataset.where(dataset["date"] >= 0, drop=True)

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

                IWP = IWP.where(IWP >= 0).fillna(0)
                TIWP = IWP * CF_ice

                if masked:
                    ds["CF_zonal_mean_masked"] += CF.sum(
                        dim=["lon", "date"], skipna=True
                    )
                    ds["TIWP_zonal_mean_masked"] += TIWP.sum(
                        dim=["lon", "date"], skipna=True
                    )
                    CF_zonal_mean_masked_count += (CF >= 0).count(dim=["lon", "date"])
                    TIWP_zonal_mean_masked_count += (TIWP >= 0).count(
                        dim=["lon", "date"]
                    )
                else:
                    ds["CF_zonal_mean_unmasked"] += CF.sum(
                        dim=["lon", "date"], skipna=True
                    )
                    ds["TIWP_zonal_mean_unmasked"] += TIWP.sum(
                        dim=["lon", "date"], skipna=True
                    )
                    CF_zonal_mean_unmasked_count += (CF >= 0).count(dim=["lon", "date"])
                    TIWP_zonal_mean_unmasked_count += (TIWP >= 0).count(
                        dim=["lon", "date"]
                    )

                dataset.close()

        lat_reversed = ds["lat"][::-1]  # lats were wrongly saved, need to be reversed

    ds["CF_zonal_mean_unmasked"] /= CF_zonal_mean_unmasked_count
    ds["TIWP_zonal_mean_unmasked"] /= TIWP_zonal_mean_masked_count
    ds["CF_zonal_mean_masked"] /= TIWP_zonal_mean_masked_count
    ds["TIWP_zonal_mean_masked"] /= TIWP_zonal_mean_masked_count

    ds = ds.assign_coords(lat=lat_reversed)

    return ds


def interpolate_mask(mask_ds, original_ds):
    """
    Interpolate the mask dataset onto modis dataset coordinates.
    Used one-time to create the mask 'mask_24_for_modis.nc'.
    """
    interpolated_mask = mask_ds.astype(int).interp(
        lat=original_ds.lat, lon=original_ds.lon, method="nearest"
    )
    return interpolated_mask


if __name__ == "__main__":
    modis_data_dir = "/scratch/ccic_record/data/modis/"
    mask = xr.open_dataset("../mask_24_for_modis.nc")

    process_global_dist = False
    process_time_series = False
    process_zonal_mean = True

    if process_global_dist:
        global_dist = process_global_distributions(modis_data_dir, mask)
        global_dist.to_netcdf("modis_global_distribution_cf_tiwp.nc")

    if process_time_series:
        df_unmasked = process_monthly_means(modis_data_dir, mask)
        df_unmasked.set_index("date", inplace=True)
        ds = df_unmasked.to_xarray().sortby("date")
        ds.to_netcdf("modis_cf_tiwp_time_series.nc")

    if process_zonal_mean:
        zonal_mean = process_zonal_means(modis_data_dir, mask)
        zonal_mean.to_netcdf("modis_zonal_mean_cf_tiwp.nc")
