"""
    Dummy conftest.py for dryspy.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest, os, sys

sys.path.insert(0, os.path.abspath("../"))

from dryspy.co_wdetect import canny_otsu_water_detect


config = {
    "edge_detection": {
        "histogram_scale_maxvalue": 100,
        "histogram_scale_minvalue": 87,
        "sigma": 2.5,
        "classes": 2,
        "threshold_multiplication": 1.0,
        "buffer": 4,
    },
    "data_clean": {
        "min_size": 8,
        "dilation_erosion_size": 4,
        "max_data_value": 0.60,
        "min_data_value": -0.48,
    },
}


def test_init():
    wd = canny_otsu_water_detect(config)

    wd.config["edge_detection"]["histogram_scale_maxvalue"]

    assert wd.config["edge_detection"]["histogram_scale_maxvalue"] == "100"


def test_read_detect():
    import rasterio
    import numpy as np

    wd = canny_otsu_water_detect(config)

    B8_ = rasterio.open(
        "notebooks/example/EO_Browser_images/2019-06-07-00:00_2019-06-07-23:59_Sentinel-2_L1C_B08_(Raw).tiff"
    )

    # Apply the scaling
    B8 = B8_.read() / 100000

    # Waterdetect assumes missing values are set to np.nan so we do that here
    B8 = np.where(B8 <= 0.001, np.nan, B8)

    water = wd.detect_water(
        B8[0, :, :], apply_uncertainty_filter=True, th_direction="larger"
    )

    assert water.sum() == 50206


def test_read_detect_multiple():
    import rasterio
    import numpy as np

    wd = canny_otsu_water_detect(config)

    B8_ = rasterio.open(
        "notebooks/example/EO_Browser_images/2019-06-07-00:00_2019-06-07-23:59_Sentinel-2_L1C_B08_(Raw).tiff"
    )
    B11_ = rasterio.open(
        "notebooks/example/EO_Browser_images/2019-06-07-00:00_2019-06-07-23:59_Sentinel-2_L1C_B11_(Raw).tiff"
    )
    B3_ = rasterio.open(
        "notebooks/example/EO_Browser_images/2019-06-07-00:00_2019-06-07-23:59_Sentinel-2_L1C_B03_(Raw).tiff"
    )
    B4_ = rasterio.open(
        "notebooks/example/EO_Browser_images/2019-06-07-00:00_2019-06-07-23:59_Sentinel-2_L1C_B04_(Raw).tiff"
    )
    # Apply the scaling
    B8 = B8_.read() / 100000
    B11 = B11_.read() / 100000
    B3 = B3_.read() / 100000
    B4 = B4_.read() / 100000

    # Waterdetect assumes missing values are set to np.nan so we do that here
    B8 = np.where(B8 <= 0.001, np.nan, B8)
    B3 = np.where(B3 <= 0.001, np.nan, B3)
    B11 = np.where(B11 <= 0.001, np.nan, B11)
    B4 = np.where(B11 <= 0.001, np.nan, B4)

    # Next we calculate nwdi from swir and nir
    ndwi_swir = (B3 - B11) / (B3 + B11)
    ndvi = (B8 - B4) / (B4 + B8)

    # Waterdetect assumes missing values are set to np.nan so we do that here
    B8 = np.where(B8 <= 0.001, np.nan, B8)

    water = wd.detect_water(
        B8[0, :, :],
        apply_uncertainty_filter=True,
        apply_secondary_filter=True,
        max_data=ndvi[0, :, :],
        min_data=ndwi_swir[0, :, :],
        th_direction="smaller",
    )

    assert water.sum() == 1009313
