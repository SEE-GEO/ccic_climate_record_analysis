"""
This script downloads CloudSat/Calipso data used for the analysis of the CCIC
climate record.
"""

from pysftp import Connection, SSHException
import os
from datetime import datetime, timedelta
from getpass import getpass


def download_files_sftp(host, user, remote_paths, local_path, key=None, pw=None):
    """
    Downloads a list of files from an SFTP server.

    Args:
        host: URL of the sftp server
        user: The sftp user name
        remote_paths: List of the paths to download.
        local_path: The local path to which to download the files.
        key: Key to use for authentication.
        pw: Password to use for autentication.

    Return:
        A list of the paths that were downloaded.
    """
    # Connect to the SFTP server
    try:
        with Connection(host, username=user, private_key=key, password=pw) as sftp:
            ok_paths = []
            for i, remote_path in enumerate(remote_paths):
                try:
                    # Change to the remote directory
                    sftp.chdir(remote_path)
                    # List files in the remote directory
                    file_list = sorted(sftp.listdir())
                    ok_paths.append(i)
                except:
                    file_list = []
                # assert len(file_list) > 0, f'No files found at remote path {remote_path}'

                for file_name in file_list:

                    remote_file_path = os.path.join(remote_path, file_name)
                    local_file_path = os.path.join(local_path, file_name)

                    # Download file
                    sftp.get(remote_file_path, local_file_path)

                    # print(f'Downloaded: {file_name}')
                print(
                    f"Downloaded files for path {i+1}/{len(remote_paths)}: {remote_path}"
                )
    except SSHException as e:
        ok_paths = []
        print("Fail to connect to remote path", e)
    return ok_paths


def download_dardar(
    credentials: dict,
    year: int,
    ordinal_days: list[int],
    destination: str,
):
    """
    Downloads DARDAR files for a list of days.

    Args:
        credentials: A dictionary containign the credentials for the ICARE server.
        year: The year from which to download the files.
        ordinal_days: The julian days for which to download the files.
        destination: The folder to which to write the files.

    Return:
        A list of the downloaded files.
    """
    remote_paths = []
    for day in ordinal_days:
        # Create datetime from year and ordinal day
        date = datetime.strptime(f"{year}-{day}", "%Y-%j")
        # Format datetime as string with format 'yyyy_mm_dd'
        date = date.strftime("%Y_%m_%d")

        remote_path = os.path.join(
            "/SPACEBORNE/CLOUDSAT/DARDAR-CLOUD.v3.10/", str(year), date
        )
        remote_paths.append(remote_path)

    ok = download_files_sftp(
        credentials["host"],
        credentials["user"],
        remote_paths,
        destination,
        pw=credentials["password"],
    )
    return ok


def download_2cice(
    credentials: dict,
    year: int,
    ordinal_days: list[int],
    destination: str,
):
    """
    Download 2C-ICE files for a list of Julian days for a given year from
    the CloudSat DPC.

    Args:
        credentials: A dictionary containing the user credentials for
            the CloudSat DPC.
        year: The year
        ordinal_days: The Julian days for which to download the files.
        destination: The local path to which to download the files.

    Return:
        A list of the downloaded files.
    """
    remote_paths = []
    for day in ordinal_days:
        remote_path = f"/Data/2C-ICE.P1_R05/{year:04}/{day:03}"
        remote_paths.append(remote_path)

    ok = download_files_sftp(
        credentials["host"],
        credentials["user"],
        remote_paths,
        destination,
        key=credentials["password"],
    )
    return ok


def download_cldclass(
    credentials: dict,
    year: int,
    ordinal_days: list[int],
    destination: str,
):
    """
    Download 2B-CLDCLASS files for a list of Julian days for a given year from.

    Args:
        credentials: A dictionary containing the user credential
        year: The year
        ordinal_days: The Julian days for which to download the files.
        destination: The local path to which to download the files.

    Return:
        A list of the downloaded files.

    """
    remote_paths = []
    for day in ordinal_days:
        remote_path = f"/Data/2B-CLDCLASS.P1_R05/{year:04}/{day:03}"
        remote_paths.append(remote_path)

    ok = download_files_sftp(
        credentials["host"],
        credentials["user"],
        remote_paths,
        destination,
        key=credentials["password"],
    )
    return ok


def download_data(dataset: str, cred: dict, date: datetime, basedir="./"):
    """
    Downloads DARDAR, 2C-ICE, or 2B-CLDCLASS data.

    Args:
        dataset: The name of the dataset to download. Should be one of
            'dardar', '2cice', 'cldclass'.
        cred: The user credential to use for authentication.
        date: A date object specifying the date for which to download the data.
        basedir: The data folder to which to download the data.

    Return:
        A list of the downloaded files.

    """
    output_dirs = {
        "dardar": "DARDAR/dardar3.1",
        "2cice": "2C-ICE",
        "cldclass": "2B-CLDCLASS",
    }

    assert dataset in output_dirs.keys(), f"{dataset}, is not a valid dataset"

    year = date.year
    mon = date.month
    day = date.day
    ordinal_day = date.timetuple().tm_yday

    # path to store downloaded data
    local_path = os.path.join(
        basedir, output_dirs[dataset], str(year), str(mon).zfill(2), str(day).zfill(2)
    )
    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    # Download data
    if dataset == "dardar":
        day_ok = download_dardar(cred, year, [ordinal_day], local_path)
    elif dataset == "2cice":
        day_ok = download_2cice(cred, year, [ordinal_day], local_path)
    else:
        day_ok = download_cldclass(cred, year, [ordinal_day], local_path)

    # remove local path if no data was found
    if len(day_ok) == 0:
        try:
            os.rmdir(local_path)
            print(f"No data found for {start_date.date()}")
        except OSError as e:
            print(f"Error: {local_path} : {e.strerror}")


if __name__ == "__main__":

    basedir = "./"

    # SFTP server credentials for downloading DARDAR
    icare_cred = {
        "host": "sftp.icare.univ-lille.fr",
        "user": "hannahallborn",
        "password": getpass("Icare password: "),
    }
    # SFTP server credentials for downloading 2C-ICE and 2B-CLDCLASS
    cloudsat_cred = {
        "host": "www.cloudsat.cira.colostate.edu",
        "user": "hallbornATchalmers.se",
        "password": "PATH/TO/KEY",
    }

    start_date = datetime(2017, 1, 1)
    end_date = datetime(2017, 1, 2)

    while start_date <= end_date:

        # Download DARDAR
        download_data("dardar", icare_cred, start_date, basedir)

        # Download 2C-ICE
        download_data("2cice", cloudsat_cred, start_date, basedir)

        # Download 2B-CLDCLASS
        download_data("cldclass", cloudsat_cred, start_date, basedir)

        start_date += timedelta(days=1)
