"""
Extract CloudSat tracks.

This script extract the time and coordinates of the CloudSat measurements for
all files available in the current environment and writes them to the output
path provided as command line argument.
"""
import sys
from pathlib import Path

from tqdm import tqdm

from pansat.environment import get_index
from pansat.products.satellite.cloudsat import l2b_cldclass_lidar

index = get_index(l2b_cldclass_lidar)
index_data = index.data.load()
files = list(set(index_data["local_path"]))

output_path = Path(sys.argv[1])

for fle in tqdm(files):
    try:
        data = l2b_cldclass_lidar.open(fle)[["latitude", "longitude", "time", "surface_elevation"]]
    except:
        print(f"Failed loading file {fle}.")
        continue
    fname = Path(fle).name
    parts = fname.split("_")[:2]
    new_name = "_".join(parts) + ".nc"
    data.to_netcdf(output_path / new_name, encoding = {
        "latitude": {"dtype": "float32", "zlib": True},
        "longitude": {"dtype": "float32", "zlib": True},
        "time": {"dtype": "float32", "zlib": True},
    })

