from datetime import datetime
from pathlib import Path

from pansat import TimeRange
from pansat.products.satellite.ncei import isccp_hgm

start_time = datetime(1983, 1, 1)
end_time = datetime(2023, 1, 1)

destination = Path("/scratch/ccic_record/data/isccp")
recs = isccp_hgm.download(TimeRange(start_time, end_time), destination=destination)
