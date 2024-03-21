import argparse
import datetime
import functools
import multiprocessing
from pathlib import Path
import subprocess
import shlex
import tempfile

import fsspec
import numpy as np
import tqdm
import xarray as xr


# Variables to download
VARIABLES = [
    "cloud_phase",
    "cloud_probability",
    "cloud_fraction",
    "cld_cwp_dcomp",
    "scan_line_time",
    "cld_opd_dcomp"
]

# S3 parameters
BUCKET_PARENT = "noaa-cdr-patmosx-radiances-and-clouds-pds/data"


def apply_custom_filename(file: str | Path) -> str:
    """Applies a custom filename pattern

    Args:
        file: the path to the file

    Returns:
        The filename with the custom pattern
    """
    return Path(file).name.replace(".nc", "_v2.zarr")


def exists_remote(host: str, path: Path | str) -> bool:
    """Test if a file exists at path on a host accessible with SSH.

    Adapted from https://stackoverflow.com/a/14392472.

    Args:
        host: the ssh alias, as configured in $HOME/.ssh/config
        path: path in the host to the file

    Returns:
        True if the file exists, False otherwise

    Raises:
        Exception: if the SSH connection failed
    """
    status = subprocess.call(["ssh", host, "test -f {}".format(shlex.quote(str(path)))])
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception("SSH failed")


def process_file(file: str, destination_dir: Path, ssh: str = None) -> None:
    """Function to encapsulate the processing of one file

    Args:
        file: path to the file in the S3 bucket
        destination_dir: where to write the file to disk
        ssh: the ssh host, as configured in $HOME/.ssh/config
    """
    # Create fs
    fs = fsspec.filesystem("s3", anon=True)
    with fs.open(f"s3://{file}") as handle:
        ds = xr.open_dataset(handle, engine="h5netcdf")[VARIABLES].load()
        # See PATMOS-x_documentation_TIWP.ipynb for understanding cld_iwp_dcomp
        cld_iwp_dcomp = xr.where(
            np.isfinite(ds.cld_opd_dcomp),
            1,
            np.nan
        ) * xr.where(
            ds.cloud_phase == 4, # ice phase
            ds.cld_cwp_dcomp,
            0
        )
        cld_iwp_dcomp.attrs = ds.cld_cwp_dcomp.attrs
        cld_iwp_dcomp.attrs['long_name'] = 'ice water path from DCOMP'
        # Clean dataset
        ds = ds.drop_vars(['cloud_phase', 'cld_cwp_dcomp', 'cld_opd_dcomp'])
        ds['cld_iwp_dcomp'] = cld_iwp_dcomp.astype(np.float32)

        scan_line_time_data = ds.scan_line_time.data
        scan_line_time_data = np.where(
            np.isfinite(scan_line_time_data),
            scan_line_time_data.astype("timedelta64[s]").astype(float),
            -1,
        ).astype(np.int32)
        ds.scan_line_time.data = scan_line_time_data
        ds.scan_line_time.attrs["long_name"] = ds.scan_line_time.attrs[
            "long_name"
        ].replace("in fractional hours", "in seconds")
        ds.scan_line_time.attrs["comment"] = "converted to seconds, -1: invalid data"

        fname_dst = apply_custom_filename(file)
        if ssh is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                dst_path = Path(tmpdir) / fname_dst
                ds.to_zarr(dst_path)
                subprocess.run(["scp", "-r", dst_path, f"{ssh}:{destination_dir}"])
        else:
            ds.to_zarr(destination_dir / fname_dst)


def run(files: list, n_proc: int, process_file_partial: functools.partial) -> None:
    """Wrapper for executing the download and processing of the files in parallel.

    Args:
        files: list with the file paths in the S3 buckets to process
        n_proc: how many processes to use
        process_file_partial: process_file function with only `file` as parameter
    """
    with multiprocessing.get_context("spawn").Pool(processes=n_proc) as pool:
        return list(
            tqdm.tqdm(
                pool.imap(process_file_partial, files),
                total=len(files),
                dynamic_ncols=True,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        required=True,
        type=lambda x: datetime.datetime.strptime(x, "%Y%m%d"),
        help="from which date (YYYYMMDD) download PATMOS-x data",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda x: datetime.datetime.strptime(x, "%Y%m%d"),
        help="until which date (YYYYMMDD) download PATMOS-x data",
    )
    parser.add_argument(
        "--destination", required=True, type=Path, help="where to place the files"
    )
    parser.add_argument(
        "--ssh",
        type=str,
        help=(
            "machine to copy the data to over ssh, "
            "as configured in $HOME/.ssh/config "
            "(requires passwordless ssh key authentication)"
        ),
    )
    parser.add_argument(
        "--n_proc", default=1, type=int, help="how many parallel processes to use"
    )

    args = parser.parse_args()

    # Create fs object to find files
    fs = fsspec.filesystem("s3", anon=True)

    # Glob files
    files_glob = []
    for year in range(args.start.year, args.end.year + 1):
        files_glob = files_glob + fs.glob(f"{BUCKET_PARENT}/{year}/**/*nc")

    files_glob = sorted(files_glob)

    # Filter out files not in the time range
    files = []
    for f in files_glob:
        f_date = datetime.datetime.strptime(f.split("_")[-2], "d%Y%m%d")
        if (args.start <= f_date) and (f_date <= args.end):
            files.append(f)

    # Filter out files already present in destination
    n_all_files = len(files)
    new_files = []
    failed_checks = []
    for f in files:
        fname = apply_custom_filename(Path(f).name)
        if args.ssh:
            try:
                if not exists_remote(args.ssh, args.destination / fname):
                    new_files.append(f)
            except:
                failed_checks.append(f)
        else:
            if not (args.destination / fname).exists():
                new_files.append(f)
    files = new_files
    n_new_files = len(files)
    message = f"{n_new_files} of {n_all_files} not present in destination"
    if len(failed_checks) > 0:
        message = f"{message}, but {len(failed_checks)} files could not be checked"
    print(message)

    process_file_partial = functools.partial(
        process_file, destination_dir=args.destination, ssh=args.ssh
    )

    run(files, args.n_proc, process_file_partial)
