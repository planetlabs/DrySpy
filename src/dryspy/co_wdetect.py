"""
***************************************************************************
Copyright Planet Labs PBC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***************************************************************************
"""

from skimage import morphology
import configparser
from skimage import exposure
import numpy as np
from skimage import feature
from skimage import measure
from skimage.filters import threshold_multiotsu
from skimage.morphology import square
from scipy.stats import kurtosis, skew
import diptest
import pandas as pd


def s_curve(X, a=0.0, b=1.0, c=1.0):
    """s_curve function, transform data into a range

    Input
    -----
        - X input array
        - C determines the steepness or "stepwiseness" of the curve.
          The higher C the sharper the function. A negative C reverses the function.
        - b determines the amplitude of the curve
        - a determines the centre level (default = 0)

    Output
    ------
        - transformed array
    """
    s = 1.0 / (b + np.exp(-c * (X - a)))

    return s


class canny_otsu_water_detect:
    def __init__(self, config, prior_water_occurrence=None):
        """Water detection using canny edge detection and otsu thresholding

        Parameters
        ----------
        config : configparser.ConfigParser or dict
            configparser object with values for the configurable items. If set to None
            default values will be used. If it is specified all the expected sections
            should be present: [edge_detection], [data_clean]. It can be specified as an
            dictionary or a configparser object
        prior_water_occurrence: 2d ndarray floats (optional)
            should have the same dimensions and the input maps used after initialisation
        """

        if config is None:
            self.config = configparser.ConfigParser(allow_no_value=True)
            self.config.read_dict(
                {
                    "edge_detection": {
                        "histogram_scale_maxvalue": 25,
                        "sigma": 2.0,
                        "classes": 3,
                        "threshold_multiplication": 1.0,
                        "buffer": 2,
                    },
                    "data_clean": {"min_size": 6, "dilation_erosion_size": 6},
                }
            )
        else:
            if isinstance(config, dict):
                self.config = configparser.ConfigParser(allow_no_value=True)
                self.config.read_dict(config)
            else:
                self.config = config  # configparser

        self.canny_buffedges = None
        self.canny_edges = None
        self.thresholds = None
        self.water_canny_otsu = None
        self.water_sec = None
        self.water_uncert = None
        self.water_final = None
        self.invalid = None
        self.data = None
        self.peak = None
        """stats 
        Statistics associated with the water detection
        
        """
        self.stats = {}
        self.count = 0
        self.wat_occur = None
        self.avguncertainty = None
        self.prior_wateroccurence = prior_water_occurrence

        self.weight_of_prior = self.config["data_clean"].getint(
            "weight_of_prior", fallback=100
        )  # 100 days
        self.u_pixel = (
            None  # Pixel based relative uncertainty (1 = certain, 0 = very uncertain)
        )
        self.u_pixel_distance = None
        self.u = None  # otsu thresholding estimated algorithm uncertainty
        self.u1 = None  # uncertainty component
        self.u2 = None  # uncertainty component
        self.u3 = None  # uncertainty component

    def _morphology_clean(self, watertoclean: np.array):
        """removes small areas and performs a dilation followed by an erosion
        to clean up the edges

        Parameters
        ----------
        watertoclean : numpy array (2d)
            incoming binary map

        Returns
        -------
        numpy array 2d
            outgoing (cleaned) binary map
        """
        # Remove small objects
        water_cleaned = morphology.remove_small_objects(
            watertoclean.copy(),
            min_size=self.config["data_clean"].getint("min_size", fallback=6),
        )
        # Do dilation and erosion to get rid of small gaps and connect objects that
        # are very close to each other
        de_size = self.config["data_clean"].getint("dilation_erosion_size", fallback=6)
        water_cleaned = morphology.binary_dilation(
            water_cleaned, footprint=np.ones((de_size, de_size))
        )
        water_cleaned = morphology.binary_erosion(
            water_cleaned, footprint=np.ones((de_size, de_size))
        )

        return water_cleaned

    def _canny_edges(self, data: np.array):
        """Apply canny edge detection after rescaling between percentiles
        percentile (default 0 and 25). Next apply a buffer and determine threshold with otsu

        Parameters
        ----------
        data : 2dndarray of floats
            data to use in edge detection
        """

        # Get histogram scaling parameters
        max = self.config["edge_detection"].getint(
            "histogram_scale_maxvalue", fallback=25
        )
        min = self.config["edge_detection"].getint(
            "histogram_scale_minvalue", fallback=0
        )
        # Scale data
        L, H = np.percentile(data[~np.isnan(data)], (min, max))
        data_rescale = exposure.rescale_intensity(data, in_range=(L, H))
        data_rescale[np.isnan(data)] = np.nan

        # Do edge detection on rescaled data
        self.canny_edges = feature.canny(
            data_rescale,
            sigma=self.config["edge_detection"].getfloat("sigma", fallback=2.0),
        )
        # Apply the buffer
        edges_buff = morphology.dilation(
            self.canny_edges,
            footprint=square(
                self.config["edge_detection"].getint("buffer", fallback=2)
            ),
        )
        # Select original data at the edge plus buffer location and determine thresholds
        data_edges = data[edges_buff]

        if data_edges.sum() == 0:
            thresholds = np.array([0.0, 0.0])
        else:
            classes = self.config["edge_detection"].getint("classes", fallback=3)
            thresholds = threshold_multiotsu(data_edges, classes=classes)
            thresholds = np.array(thresholds)

        # TDO: nr op sample points?
        # apply multiplication if configured
        self.canny_buffedges = edges_buff
        self.thresholds = thresholds.copy()
        mult = self.config["edge_detection"].getfloat(
            "threshold_multiplication", fallback=1.0
        )
        self.thresholds = self.thresholds * float(mult)

        return self.thresholds[0]

    def _apply_secundary_filter(
        self, water_in: np.array, max_data: np.array, min_data: np.array
    ):
        """Apply a secundary filter on the detected water
        if these values are not configure in the config the filter wil _NOT_ be applied.
        This can be run _after_ the detect_water method

        [data_clean]
        - max filter using max_data
        - min filter using min_date

        Parameters
        ----------
        water_in : np.array
            initial estimate of water to be filtered
        max_data : np.array
            array with  data to use above which we assume there is no water
            (typically something like an ndvi)
        min_data : np.array
            array with  data to use below which we assume there is no water
            (typically something like an ndwi)
        """
        max_data_value = self.config["data_clean"].getfloat(
            "max_data_value", fallback=-999.0
        )
        min_data_value = self.config["data_clean"].getfloat(
            "min_data_value", fallback=-999.0
        )

        self.water_sec = water_in.copy()
        if max_data_value > -998:
            filt = max_data > max_data_value
            self.water_sec[filt] = 0
        if min_data_value > -998:
            filt = min_data < min_data_value
            self.water_sec[filt] = 0

        if np.sum(self.water_sec) != np.sum(water_in):
            changed = True
        else:
            changed = False

        return self.water_sec.copy(), changed

    def _det_and_apply_uncertainty_filter(self, water_in: np.array):
        """_summary_"""

        if self.count == 1:
            self.sumuncertainty = np.nan_to_num(self.u_pixel.astype(float).copy())
        else:
            self.sumwat = self.sumwat + water_in.astype(int)
            self.sumuncertainty = np.nan_to_num(self.sumuncertainty) + np.nan_to_num(
                self.u_pixel
            )

        self.water_uncert = water_in.copy()

        # First determine average uncertainty per water body.
        min_u_area = self.config["data_clean"].getfloat("min_u_area", fallback=0)
        self.label_areas = measure.label(water_in)
        props = measure.regionprops_table(
            self.label_areas,
            self.u_pixel,
            properties=[
                "label",
                "area",
                "intensity_mean",
                "axis_major_length",
                "axis_minor_length",
            ],
        )
        props = pd.DataFrame(props)
        areas_to_filter = props.where(props["intensity_mean"] < min_u_area).dropna()
        for area in areas_to_filter.iterrows():
            self.water_uncert[area[1]["label"] == self.label_areas] = 0

        # Next filter out points uncertain points based on this timestep
        min_u_pixel = self.config["data_clean"].getfloat("min_u_pixel", fallback=0)

        self.water_uncert[self.u_pixel < min_u_pixel] = 0

        if self.prior_wateroccurence is None:
            self.wat_occur = self.sumwat / self.count
        else:
            self.wat_occur = self.sumwat / (self.count + self.weight_of_prior)

        self.avguncertainty = self.sumuncertainty / self.count

        # Filter out points based on history (can only be done if the object is
        # run multiple times)
        min_occur = self.config["data_clean"].getfloat("min_occur", fallback=0)
        self.water_uncert[self.wat_occur < min_occur] = 0

        if np.sum(self.water_uncert) != np.sum(water_in):
            changed = True
        else:
            changed = False

        return self.water_uncert.copy(), changed

    def _extra_stats(self):
        """Add some extra statistics to the stats dictionary"""
        L, H = np.percentile(
            self.wat_occur[~np.isnan(self.wat_occur)],
            (5, 95),
        )
        self.stats["diff_in_out"] = np.nanmean(
            self.data[self.wat_occur >= H]
        ) - np.nanmean(self.data[self.wat_occur <= L])
        self.label_areas = measure.label(self.water_final)
        props = measure.regionprops_table(
            self.label_areas,
            self.data,
            properties=[
                "label",
                "area",
                "intensity_mean",
                "axis_major_length",
                "axis_minor_length",
            ],
        )
        props = pd.DataFrame(props)
        props.sort_values(by=["area"], ascending=False, inplace=True)
        i = 0
        for prop in props.iterrows():
            i = i + 1
            if i <= 20:
                self.stats["wb_area_" + str(i)] = prop[1].area
                self.stats["wb_mean_" + str(i)] = prop[1].intensity_mean
                self.stats["wb_area_length_" + str(i)] = prop[1].axis_major_length
                self.stats["wb_area_length_short_" + str(i)] = prop[1].axis_minor_length
                self.stats["wb_area_label_" + str(i)] = prop[1].label

    def detect_water(
        self,
        data: np.array,
        th_direction="smaller",
        apply_secondary_filter=False,
        max_data: np.array = None,
        min_data: np.array = None,
        apply_uncertainty_filter=False,
    ):
        """Calls the canny/otsu water detection method using the 2darray in data,
        sets the output in object-wide variables, estimates relative uncertainty
        and does some post-processing.

        Parameters
        ----------
        data : 2d array of floats non valid data should be indicated as np.nan
            grid to use in water detection
        th_direction: string
            direction of threshold smaller|larger (e.g. for ndwi you would chose larger
            (water is in the high range), for nir you would chose smaller

        Returns
        -------
        2d boolean array
            water grid (1 === water, 0 == no water)
        """
        self.data = data.copy()
        if data.dtype != float:
            raise TypeError(f"data should be of type float, got: '{data.dtype}'")

        # step 1 edge detection
        self._canny_edges(data)

        # First calculate some stats for the input data
        buffdata = self.data[self.canny_buffedges].copy()
        self.stats["hist_kurtosis"] = kurtosis(buffdata[~np.isnan(buffdata)].flatten())
        self.stats["hist_skew"] = skew(buffdata[~np.isnan(buffdata)].flatten())
        if len(~np.isnan(buffdata).flatten()) > 0:
            self.stats["25"], self.stats["75"] = np.nanpercentile(
                buffdata, [25.0, 75.0]
            )
        else:
            self.stats["25"] = np.nan
            self.stats["75"] = np.nan

        if np.sum((buffdata[~np.isnan(buffdata)].flatten())) > 0:
            self.stats["diptest"] = diptest.dipstat(
                buffdata[~np.isnan(buffdata)].flatten()
            )
        else:
            self.stats["diptest"] = np.nan
        # Convert kurt and skew to an apparent uncertainty in the otsu thresholding
        self.u1 = s_curve(self.stats["hist_skew"], a=2, b=1, c=-1)
        self.u2 = s_curve(self.stats["hist_kurtosis"], a=2, b=1, c=-1)

        # the slope if the histogram at the threshold is another source of
        # uncertainty. Here we determine the slope and use the average of three points
        # TODO: also add for original hist (non sampled)
        hist = np.histogram(buffdata, bins="auto", density=True)
        difs = np.absolute(self.thresholds[0] - hist[1])
        idx = difs.argmin()  # index closest to threshold
        slopes = np.diff(hist[0])
        if np.isnan(hist[0][[0]]):
            self.slope_at_threshold = np.nan
        else:
            self.slope_at_threshold = np.nanmean(
                np.abs(slopes[idx - 1 : idx + 1]) / np.nanmean(np.diff(hist[1]))
            )
        self.stats["hist_slope"] = self.slope_at_threshold
        self.u3 = s_curve(self.slope_at_threshold, a=35000, b=1, c=-0.00007)

        # We take the mean of the three relative uncertainties as they are
        # different estimates of the same uncertainty
        self.u = (self.u1 + self.u2 + self.u3) / 3.0

        if th_direction == "larger":
            self.water_canny_otsu = data > self.thresholds[0]
        else:
            self.water_canny_otsu = data < self.thresholds[0]

        self.water_canny_otsu = self._morphology_clean(self.water_canny_otsu)

        self.water_final = self.water_canny_otsu.copy()
        # estimate pixel bases uncertainty using the distance to the threshold,
        # max out at 1
        self.u_pixel_distance = np.minimum(
            1.0,
            np.absolute(self.data - self.thresholds[0])
            / (self.stats["75"] - self.stats["25"]),
        )
        self.u_pixel = self.u_pixel_distance * self.u
        self.u_pixel[self.water_final == 0] = np.nan

        self.sec_changed = False
        self.uncert_changed = False
        if apply_secondary_filter:
            self.water_final, self.sec_changed = self._apply_secundary_filter(
                self.water_final, max_data, min_data
            )

        self.count = self.count + 1  # for temporal occurrence
        if self.count == 1:
            if self.prior_wateroccurence is None:
                self.sumwat = self.water_final.copy()
                self.wat_occur = self.sumwat.copy().astype(int)
            else:
                self.sumwat = self.prior_wateroccurence * self.weight_of_prior
                self.wat_occur = self.prior_wateroccurence

        if apply_uncertainty_filter:
            (
                self.water_final,
                self.uncert_changed,
            ) = self._det_and_apply_uncertainty_filter(self.water_final)

        self.water_final = self._morphology_clean(self.water_final)

        # fill the stats dictionary with extra statistics
        self._extra_stats()

        return self.water_final
