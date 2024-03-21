from ftplib import FTP
from  pysftp import Connection, SSHException
import os
from datetime import datetime, timedelta
from getpass import getpass

def download_files_ftp(host, user, remote_paths, local_path, pw):
    """
    remote_paths: list of remote directories
    """
    # Connect to the FTP server
    with FTP(host) as ftp:
        # Login to the FTP server
        ftp.login(user, pw)
        
        ok_paths = []
        for i,remote_path in enumerate(remote_paths):
            try:
                # Change to the remote directory
                ftp.cwd(remote_path)
                # List files in the remote directory
                file_list = sorted(ftp.nlst())
                ok_paths.append(i)
            except:
                file_list = []
            #assert len(file_list) > 0, f'No files found at remote path {remote_path}'
            for file_name in file_list:

                remote_file_path = os.path.join(remote_path, file_name)
                local_file_path = os.path.join(local_path, file_name)

                # Download file
                with open(local_file_path, 'wb') as local_file:
                    ftp.retrbinary(f'RETR {remote_file_path}', local_file.write)

                #print(f'Downloaded: {file_name}')
            print(f"Downloaded files for path {i+1}/{len(remote_paths)}: {remote_path}")
    return ok_paths

def download_files_sftp(host, user, remote_paths, local_path, key=None, pw=None):
    """
    remote_paths: list of remote directories
    """
    # Connect to the SFTP server
    try:
        with Connection(host, username=user, private_key=key, password=pw) as sftp:
            ok_paths = []
            for i,remote_path in enumerate(remote_paths):
                try:
                    # Change to the remote directory
                    sftp.chdir(remote_path)
                    # List files in the remote directory
                    file_list = sorted(sftp.listdir())
                    ok_paths.append(i)
                except:
                    file_list = []
                #assert len(file_list) > 0, f'No files found at remote path {remote_path}'

                for file_name in file_list:

                    remote_file_path = os.path.join(remote_path, file_name)
                    local_file_path = os.path.join(local_path, file_name)

                    # Download file
                    sftp.get(remote_file_path, local_file_path)

                    #print(f'Downloaded: {file_name}')
                print(f"Downloaded files for path {i+1}/{len(remote_paths)}: {remote_path}")
    except SSHException as e:
        ok_paths = []
        print("Fail to connect to remote path", e)
    return ok_paths

def download_dardar(
        credentials :   dict,
        year        :   int, 
        ordinal_days:   list[int], 
        destination :   str,
    ):
    remote_paths = []
    for day in ordinal_days:
        # Create datetime from year and ordinal day
        date = datetime.strptime(f"{year}-{day}", "%Y-%j")
        # Format datetime as string with format 'yyyy_mm_dd'
        date = date.strftime("%Y_%m_%d")

        remote_path = os.path.join(
            '/SPACEBORNE/CLOUDSAT/DARDAR-CLOUD.v3.10/', str(year), date
        )
        remote_paths.append(remote_path)

    ok = download_files_sftp(
        credentials['host'], 
        credentials['user'],  
        remote_paths, 
        destination,
        pw=credentials['password'],
    )
    return ok

def download_2cice(
        credentials :   dict,
        year        :   int, 
        ordinal_days:   list[int], 
        destination :   str,
    ):
    remote_paths = []
    for day in ordinal_days:
        remote_path = f"/Data/2C-ICE.P1_R05/{year:04}/{day:03}"
        remote_paths.append(remote_path)

    ok = download_files_sftp(
        credentials['host'], 
        credentials['user'],  
        remote_paths, 
        destination,
        key=credentials['password'],
    )
    return ok

def download_cldclass(
        credentials :   dict,
        year        :   int, 
        ordinal_days:   list[int], 
        destination :   str,
    ):
    remote_paths = []
    for day in ordinal_days:
        remote_path = f"/Data/2B-CLDCLASS.P1_R05/{year:04}/{day:03}"
        remote_paths.append(remote_path)

    ok = download_files_sftp(
        credentials['host'], 
        credentials['user'], 
        remote_paths, 
        destination,
        key=credentials['password'],
    )
    return ok


def download_data(dataset : str, cred  :  dict, date : datetime, basedir='./'):

    output_dirs = {
        'dardar'    :   'DARDAR/dardar3.1',
        '2cice'     :   '2C-ICE',
        'cldclass'  :   '2B-CLDCLASS',
    }
    
    assert dataset in output_dirs.keys(), f'{dataset}, is not a valid dataset'

    year = date.year
    mon = date.month
    day = date.day
    ordinal_day = date.timetuple().tm_yday
    
    # path to store downloaded data
    local_path = os.path.join(basedir, output_dirs[dataset], str(year), str(mon).zfill(2), str(day).zfill(2)) 
    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Download data
    if dataset == 'dardar':
        day_ok = download_dardar(cred, year, [ordinal_day], local_path)
    elif dataset == '2cice':
        day_ok = download_2cice(cred, year, [ordinal_day], local_path)
    else:
        day_ok = download_cldclass(cred, year, [ordinal_day], local_path)

    # remove local path if no data was found
    if len(day_ok) == 0:
        try:
            os.rmdir(local_path)
            print(f'No data found for {start_date.date()}')
        except OSError as e:
            print(f'Error: {local_path} : {e.strerror}')



if __name__ == "__main__":

    basedir = './'

    # SFTP server credentials for downloading DARDAR
    icare_cred = {
        'host'      :   'sftp.icare.univ-lille.fr',
        'user'      :   'hannahallborn',
        'password'  :   getpass('Icare password: '),
    }
    # SFTP server credentials for downloading 2C-ICE and 2B-CLDCLASS
    cloudsat_cred = {
        'host'      :   'www.cloudsat.cira.colostate.edu',
        'user'      :   'hallbornATchalmers.se',
        'password'  :   'PATH/TO/KEY',
    }

    start_date = datetime(2017, 1, 1)
    end_date = datetime(2017, 1, 2)

    while start_date <= end_date:

        # Download DARDAR
        download_data('dardar', icare_cred, start_date, basedir)

        # Download 2C-ICE
        download_data('2cice', cloudsat_cred, start_date, basedir)
        
        # Download 2B-CLDCLASS
        download_data('cldclass', cloudsat_cred, start_date, basedir)

        start_date += timedelta(days=1)





