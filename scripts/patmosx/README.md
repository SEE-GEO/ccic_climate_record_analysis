# Download and pre-process PATMOS-x data

Scripts:

- `PATMOS-x_download.py`

  Downloads PATMOS-x v6.0 data and applies the processing described in extract IWP. Creates a Zarr file with variables `cld_iwp_dcomp` (IWP, computed following [`notebooks/PATMOS-x_documentation_TIWP.ipynb`](../../notebooks/patmosx/PATMOS-x_documentation_TIWP.ipynb)), `cloud_fraction`, `cloud_probability`, `scan_line_time`. It was designed to be executed on the local cluster. Related: [issue #2](https://github.com/SEE-GEO/ccic_climate_record_analysis/issues/2#issuecomment-1895586527)

- `PATMOS-x_hourlymonthlymean.py`: Compute pixel-wise monthly means separated by hour of day using the data prepared with `PATMOS-x_download.py`.

- `PATMOS-x_monthlymean.py`: Compute monthly means from the hourly monthly means with the data prepared with `PATMOS-x_hourlymonthlymean.py`.
