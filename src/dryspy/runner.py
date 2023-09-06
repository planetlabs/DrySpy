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

import rich_click as click
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import configparser
from skimage import morphology
from skimage import measure
import logging
from pathlib import Path
import rasterio

plt.style.use("ggplot")

from dryspy.rw_utils import *
from dryspy.co_wdetect import canny_otsu_water_detect

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO
)

click.rich_click.USE_MARKDOWN = True
np.seterr(divide="ignore", invalid="ignore")


# TODO: all bands converted to float on reading and add np.nan


class dryspy:
    def __init__(
        self,
        dryspy_config_file,
        file_type,
        clip_file=None,
        prior_water_occurrence=None,
    ):
        """call water detection for a date range and derive the dryspy statistics."""

        self.filetype = file_type
        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read(dryspy_config_file)
        self.clip_file = self.config["area"].get("clip_file", fallback=clip_file)
        self.nir = None
        self.blue = None
        self.green = None
        self.red = None
        self.rgb = None
        self.gray = None
        self.ndwi = None
        self.ndvi = None
        self.invalid = None  # invalid (missing) data in the image
        self.out_meta = None  # meta data for the output file (based on clip file)
        self.water_canny_otsu = None
        self.water_th = None
        self.th_nir = None
        self.th_ndwi = None
        self.canny_buffedges = None
        self.stats = []
        self.thresholds = None
        self.date = None
        self.label_areas = None
        self.qa = None
        self.count = 0
        self.co_waterdetection = None
        # initialize canny/otsu object
        # self.prior = prior_water_occurrence.copy()
        self.co_waterdetection = canny_otsu_water_detect(
            self.config, prior_water_occurrence
        )
        self.clonemap = None

    def save_geotif_results(self, output_folder="."):
        output_folder_tif = output_folder + "/TIF/"
        fnameinfile = str(self.date) + ".tif"
        self.out_meta["count"] = "4"

        if self.filetype == "PF":
            with rasterio.open(
                fp=os.path.join(output_folder_tif + "/PF-SR/", fnameinfile),
                mode="w",
                **self.out_meta,  # outputpath_name
            ) as dst:
                dst.write_band(1, self.blue * 10000)
                dst.write_band(2, self.green * 10000)
                dst.write_band(3, self.red * 10000)
                dst.write_band(4, self.nir * 10000)

            with rasterio.open(
                fp=os.path.join(output_folder_tif + "/PF-QA/", fnameinfile),
                mode="w",
                **self.out_meta_qa,  # outputpath_name
            ) as dst:
                dst.write(self.qa)

        fnamendvi = "ndvi_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fnamendvi),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.ndvi * 1000)

        fname = "avgwat_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fname),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.co_waterdetection.wat_occur * 1000)

        fname = "avguncertainty_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fname),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.co_waterdetection.avguncertainty * 1000)

        fname = "upixel_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fname),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.co_waterdetection.u_pixel * 1000)

        fname = "water_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fname),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.water_canny_otsu)

        fname = "buffedges_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fname),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.canny_buffedges)

        fname = "waterbodies_" + str(self.date) + ".tif"
        self.out_meta["count"] = "1"
        with rasterio.open(
            fp=os.path.join(output_folder_tif, fname),
            mode="w",
            **self.out_meta,  # outputpath_name
        ) as dst:
            dst.write_band(1, self.label_areas)

    def save_classification_png(self, output_folder="."):
        """save a png with a graphical overview of the classification results"""

        output_folder_png = output_folder + "/PNG/"
        pdstats = pd.DataFrame(self.stats)
        pdstats.index = pdstats["Date"]

        rgb = self.rgb.swapaxes(0, 1).copy()
        rgb = rgb.swapaxes(1, 2)
        rgb = np.ma.masked_where(rgb == 0, rgb)

        image = self.nir.copy()
        image = np.ma.masked_where(image == 0, image)
        fig, ax = plt.subplots(3, 3, figsize=(24, 16))
        if self.clip_file is not None:
            fig.suptitle(
                str(self.date) + ": " + os.path.basename(self.clip_file), fontsize=28
            )

        ax[0, 0].imshow(rgb)
        ax[0, 0].set_title("rgb")

        ax[0, 2].set_title("canny edges")
        im = ax[0, 2].imshow(image + (self.canny_buffedges * 0.5))
        plt.colorbar(im, ax=ax[0, 2])

        water_ = np.ma.masked_where(self.water_final == 0, self.water_final)
        ax[1, 2].imshow(water_, cmap="gray")
        ax[1, 2].set_title("detected water")

        ndwi = np.ma.masked_where(self.ndwi == 0, self.ndwi)
        ax[1, 0].set_title("ndwi")
        im = ax[1, 0].imshow(ndwi)
        plt.colorbar(im, ax=ax[1, 0])

        ax[0, 1].set_title("nir")
        im = ax[0, 1].imshow(image)
        plt.colorbar(im, ax=ax[0, 1])

        ndvi = np.ma.masked_where(self.ndvi == 0, self.ndvi)
        ax[1, 1].set_title("ndvi")
        im = ax[1, 1].imshow(ndvi)
        plt.colorbar(im, ax=ax[1, 1])

        wo = np.ma.masked_where(
            self.co_waterdetection.wat_occur == 0, self.co_waterdetection.wat_occur
        )
        ax[2, 1].set_title("water occurence")
        im = ax[2, 1].imshow(wo)
        plt.colorbar(im, ax=ax[2, 1])
        u = np.ma.masked_where(
            self.co_waterdetection.u_pixel == 0, self.co_waterdetection.u_pixel
        )
        ax[2, 0].set_title("uncertainty")
        im = ax[2, 0].imshow(u, vmin=0.5, vmax=0.9)
        plt.colorbar(im, ax=ax[2, 0])

        counts, bins = np.histogram(
            self.nir[~np.isnan(self.nir)], bins="auto", density=True
        )
        counts1, bins1 = np.histogram(
            self.nir[self.co_waterdetection.canny_buffedges], density=True, bins="auto"
        )
        ax[2, 2].hist(bins[:-1], bins, weights=counts)
        ax[2, 2].hist(bins1[:-1], bins1, weights=counts1)
        ax[2, 2].set_title("nir histogram")
        ax[2, 2].axvline(self.thresholds[0], color="r")
        ax[2, 2].axvline(self.th_nir, color="g")
        ax[2, 2].set_title("Histogram and threshold (red=otsu)")

        fname = str(self.date)
        plt.savefig(os.path.join(output_folder_png, fname))
        plt.close()

    def cp_PF_from_bucket_to_local(self, date, PFpath, QApath, tmppath="."):
        """temporary function to read from bucket using gsutil"""

        fname = str(date) + ".tif"
        QA_tmppath = tmppath + "/PF-QA/" + fname
        PF_tmppath = tmppath + "/PF-SR/" + fname

        if not os.path.exists(PF_tmppath):
            os.system("gsutil cp " + PFpath + " " + '"' + PF_tmppath + '"')
        if not os.path.exists(QA_tmppath):
            os.system("gsutil cp " + QApath + " " + '"' + QA_tmppath + '"')

        return PF_tmppath, QA_tmppath

    def read_PS(self, psfile, udmfile, xmlfile, date):
        (
            self.red,
            self.green,
            self.blue,
            self.nir,
            self.invalid,
            self.out_meta,
            self.out_meta_qa,
            self.qa,
            src,
        ) = read_and_correct_PS(
            psfile, udmfile, xmlfile, clonemap=self.clonemap, clip_file=self.clip_file
        )

        if self.clonemap is None:
            self.clonemap = src

        self.date = date
        self.ndwi = (self.green - self.nir) / (self.green + self.nir)
        self.ndvi = (self.nir - self.red) / (self.nir + self.red)
        self.rgb = np.array([self.red * 20, self.green * 30, self.blue * 15])

    def read_PF(self, pffile, qafile, date, tmp_folder="/tmp"):
        (
            self.red,
            self.green,
            self.blue,
            self.nir,
            self.invalid,
            self.out_meta,
            self.out_meta_qa,
            self.qa,
        ) = read_and_correct_PF(pffile, qafile, clip_file=self.clip_file)

        self.gray = 0.114 * self.blue + 0.299 * self.red + 0.587 * self.green
        self.date = date
        self.ndwi = (self.green - self.nir) / (self.green + self.nir)
        self.ndvi = (self.nir - self.red) / (self.nir + self.red)
        self.rgb = np.array([self.red * 6, self.green * 6, self.blue * 6])

    def read_S2(self, s2file, date, tmp_folder="/tmp"):
        (
            self.red,
            self.green,
            self.blue,
            self.nir,
            self.invalid,
            self.out_meta,
            self.out_meta_qa,
            self.qa,
        ) = read_and_correct_S2(s2file, clip_file=self.clip_file)

        self.gray = 0.114 * self.blue + 0.299 * self.red + 0.587 * self.green
        self.date = date
        self.ndwi = (self.green - self.nir) / (self.green + self.nir)
        self.ndvi = (self.nir - self.red) / (self.nir + self.red)
        self.rgb = np.array([self.red * 5, self.green * 5, self.blue * 5])

    def detect_water(self, data):
        """calls the generic water detection, sets variable and adds result of trivial method"""
        self.count = self.count + 1
        # Using percentiles
        tmp = self.nir.copy()
        tmp[tmp == 0] = np.nan
        th_nir = self.config["nir_and_ndwi"].getfloat("nir_th", fallback=-999)
        if th_nir == -999:
            th_nir = np.nanpercentile(
                tmp, self.config["nir_and_ndwi"].getfloat("nir_th_perc", fallback=5)
            )
        tmp = self.ndwi.copy()
        tmp[tmp == 0] = np.nan
        th_ndwi = self.config["nir_and_ndwi"].getfloat("ndwi_th", fallback=-999)
        if th_ndwi == -999:
            th_ndwi = np.nanpercentile(
                tmp, self.config["nir_and_ndwi"].getfloat("ndwi_th_perc", fallback=5)
            )

        self.th_nir = th_nir
        self.th_ndwi = th_ndwi

        self.water_th = np.logical_and(self.ndwi > th_ndwi, self.nir < th_nir)

        self.water_th = morphology.remove_small_objects(
            self.water_th,
            min_size=self.config["data_clean"].getint("min_size", fallback=6),
        )

        self.co_waterdetection.detect_water(
            data.copy(),
            th_direction="smaller",
            apply_secondary_filter=True,
            max_data=self.ndvi,
            min_data=self.ndwi,
            apply_uncertainty_filter=True,
        )
        self.canny_buffedges = self.co_waterdetection.canny_buffedges.copy()
        self.thresholds = self.co_waterdetection.thresholds.copy()
        self.water_canny_otsu = self.co_waterdetection.water_canny_otsu.copy()
        self.water_uncert = self.co_waterdetection.water_uncert.copy()
        self.water_sec = self.co_waterdetection.water_sec.copy()
        self.water_final = self.co_waterdetection.water_final.copy()

    def temporal_stats(self):
        from skimage.filters import threshold_multiotsu

        """Statistics that rely on the time component, calculated once at the end"""

        # Determine wet/dry theshold based on the longest (2 if present) water bodies)
        pdstats = pd.DataFrame(self.stats)

        nr_wb = self.config["dry_detection"].getint("number_wb", fallback=3)
        tot = 0
        for nr in np.arange(0, nr_wb):
            tot = pdstats["wb_area_length_" + str(nr + 1)].fillna(0) + tot
        threshold = threshold_multiotsu(tot.values, classes=3)
        dry = tot.rolling(4, center=True).mean() < threshold[0]
        pdstats["dry_wb_length"] = dry.values

        nir_th = self.config["dry_detection"].getfloat("nir_diff_th", fallback=-0.05)
        rundry_nir = pdstats["NIR_diff"] >= nir_th
        pdstats["dry_nir_diff"] = rundry_nir.values

        # Put this back in the list of dicts
        self.stats = pdstats.to_dict("records")

    def update_stats(self):
        """statistics based on the generate open water map and the udm information
        This is a LOT of information but it may be usfull to diagnose what is going on.
        """

        # Label all water bodies so we can measure the sizes
        self.label_areas = measure.label(self.water_canny_otsu)

        stats = {}

        stats["Date"] = str(self.date)
        stats["water_th"] = self.water_th.sum()  # Number of pixels classified as water
        stats["water_canny_otsu"] = self.water_canny_otsu.sum()
        stats["water_secondary"] = self.water_sec.sum()
        stats["water_uncert"] = self.water_uncert.sum()
        stats["water"] = self.water_final.sum()
        stats["total"] = self.nir.shape[0] * self.nir.shape[1]
        stats["slope_at_th"] = self.co_waterdetection.slope_at_threshold
        # other needed to make no-water stats
        other = self.water_canny_otsu.copy()
        other[self.water_canny_otsu == 0] = 1
        other[self.invalid == 1] = 2
        other[self.water_canny_otsu == 1] = 0
        valid = self.invalid == 0

        self.nir[self.nir == 0.0] = np.nan
        self.red[self.red == 0.0] = np.nan
        self.green[self.green == 0.0] = np.nan
        self.blue[self.blue == 0.0] = np.nan
        stats["other_mean_nir"] = np.nanmean(self.nir[other == 1])
        stats["other_mean_ndwi"] = np.nanmean(self.ndwi[other == 1])
        stats["other_mean_ndvi"] = np.nanmean(self.ndvi[other == 1])
        stats["other_mean_green"] = np.nanmean(self.green[other == 1])
        stats["other_mean_blue"] = np.nanmean(self.blue[other == 1])
        stats["other_mean_red"] = np.nanmean(self.red[other == 1])
        stats["water_mean_nir"] = np.nanmean(self.nir[self.water_canny_otsu == 1])
        stats["water_mean_ndwi"] = np.nanmean(self.ndwi[self.water_canny_otsu == 1])
        stats["water_mean_ndvi"] = np.nanmean(self.ndvi[self.water_canny_otsu == 1])
        stats["water_mean_green"] = np.nanmean(self.green[self.water_canny_otsu == 1])
        stats["water_mean_blue"] = np.nanmean(self.blue[self.water_canny_otsu == 1])
        stats["water_mean_red"] = np.nanmean(self.red[self.water_canny_otsu == 1])

        L, H = np.percentile(
            self.co_waterdetection.wat_occur[
                ~np.isnan(self.co_waterdetection.wat_occur)
            ],
            (5, 95),
        )
        stats["NIR_diff"] = np.nanmean(
            self.nir[self.co_waterdetection.wat_occur >= H]
        ) - np.nanmean(self.nir[self.co_waterdetection.wat_occur <= L])

        stats["buffed_edges_points"] = np.sum(self.co_waterdetection.canny_buffedges)
        stats["valid"] = valid.sum()
        stats["invalid"] = self.invalid.sum()
        stats["water_frac_total"] = stats["water"] / stats["total"]
        stats["water_frac_valid"] = stats["water"] / stats["valid"]
        stats["tr_water_frac_total"] = stats["water_th"] / stats["total"]
        stats["tr_water_frac_valid"] = stats["water_th"] / stats["valid"]
        stats["nir_th"] = self.th_nir
        stats["ndwi_th"] = self.th_ndwi
        stats["nir_th_otsu"] = self.thresholds[0]
        stats["hist_kurtosis"] = self.co_waterdetection.stats["hist_kurtosis"]
        stats["hist_skew"] = self.co_waterdetection.stats["hist_skew"]
        stats["hist_slope"] = self.co_waterdetection.stats["hist_slope"]
        stats["diptest"] = self.co_waterdetection.stats["diptest"]
        stats["u_pixel"] = np.nanmean(
            self.co_waterdetection.u_pixel[self.water_canny_otsu == 1]
        )
        stats["u_pixel_distance"] = np.nanmean(
            self.co_waterdetection.u_pixel_distance[self.water_canny_otsu == 1]
        )
        stats["u1"] = self.co_waterdetection.u1
        stats["u2"] = self.co_waterdetection.u2
        stats["u3"] = self.co_waterdetection.u3

        if self.qa is not None:
            qa = self.qa.copy().astype(float)
            qa[qa == -999] = np.nan
            stats["image_qa"] = np.nanmean(qa[0, :, :])

        # Make statistic per water body
        bnd = [self.nir, self.green, self.blue, self.red]
        bndname = ["nir", "green", "blue", "red"]

        bndcount = 0
        for band, name in zip(bnd, bndname):
            bndcount = bndcount + 1
            props = measure.regionprops_table(
                self.label_areas,
                band,
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
                    if bndcount == 1:
                        stats["wb_area_" + str(i)] = prop[1].area
                        stats["wb_mean_" + name + "_" + str(i)] = prop[1].intensity_mean
                        stats["wb_area_length_" + str(i)] = prop[1].axis_major_length
                        stats["wb_area_length_short_" + str(i)] = prop[
                            1
                        ].axis_minor_length
                        stats["wb_area_label_" + str(i)] = prop[1].label
                    else:
                        stats["wb_mean_" + name + "_" + str(i)] = prop[1].intensity_mean

        self.stats.append(stats)


@click.command(no_args_is_help=True)
@click.option(
    "--input_folder",
    default=".",
    help="input folder with planetscope data (PS, UDM and XML), FUSION (PF) data \
    or Sentinel-2 (S2) Data (float type, downloaded from sentinel hub) ",
)
@click.option(
    "--prior_water_occurrence",
    default=None,
    help="input file (1 band tiff (0-1)) of prior water occurence",
)
@click.option(
    "--clip_file", default=None, help="geojson file used to clip the input files"
)
@click.option("--file_type", default="PS", help="type of input files: PS | PF | S2")
@click.option(
    "--output_folder",
    default="./dryspy_output",
    help="output folder, will be created if not present",
)
@click.option("--tmp_folder", default="/tmp", help="scratch folder")
@click.option(
    "--save_overview_pngs",
    "-sp",
    is_flag=True,
    default=True,
    help="Save an overview png for each date.",
)
@click.option(
    "--save_tiff_results",
    "-st",
    is_flag=True,
    default=False,
    help="Save geotif results for each date.",
)
@click.option(
    "--config_file",
    default="dryspy.ini",
    help="Config file, default is dryspy.ini",
)
@click.option("--startdate", default=None, help="only process file after this date")
@click.option("--enddate", default=None, help="only process file before this date")
def run(
    input_folder,
    prior_water_occurrence,
    output_folder,
    tmp_folder,
    save_overview_pngs,
    save_tiff_results,
    config_file,
    file_type,
    startdate,
    enddate,
    clip_file,
):
    """Run the dryspy program. Detect water in regions and calculate statistics"""
    non_matching_size = 0
    input_list = construct_input_list_from_start_and_enddate(
        input_folder, type=file_type, startdate=startdate, enddate=enddate
    )
    print(input_list)

    if prior_water_occurrence is not None:
        ds = rasterio.open(prior_water_occurrence)
        # prio = ds.read(1) * ds.scales + ds.offsets
        # for now
        prio = ds.read(1) * 0.001
    else:
        prio = None
    # initialize the detection and data object
    dryspy_obj = dryspy(config_file, file_type, clip_file, prior_water_occurrence=prio)

    # create ouput dirs if it does not exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(output_folder + "/PNG/").mkdir(parents=True, exist_ok=True)
    if file_type == "PF":
        Path(output_folder + "/TIF/PF-SR/").mkdir(parents=True, exist_ok=True)
        Path(output_folder + "/TIF/PF-QA/").mkdir(parents=True, exist_ok=True)
    if file_type == "PS" or file_type == "S2":
        Path(output_folder + "/TIF/").mkdir(parents=True, exist_ok=True)

    for infile in input_list.iterrows():
        date_obj = infile[0].date()
        logging.info("Now processing: " + str(date_obj))
        logging.info("Reading input...: " + str(infile[1]))
        if clip_file is not None:
            logging.info("clipping with: " + clip_file)
        if file_type == "PS":
            dryspy_obj.read_PS(
                infile[1]["PS"], infile[1]["UDM"], infile[1]["XML"], date_obj
            )
        if file_type == "PF":
            dryspy_obj.read_PF(
                infile[1]["PF"], infile[1]["QA"], date_obj, tmp_folder=tmp_folder
            )
        if file_type == "S2":
            dryspy_obj.read_S2(infile[1]["S2"], date_obj, tmp_folder=tmp_folder)

        if dryspy_obj.clonemap is not None:
            if (dryspy_obj.clonemap.height != dryspy_obj.nir.shape[0]) or (
                dryspy_obj.clonemap.width != dryspy_obj.nir.shape[1]
            ):
                non_matching_size = 1
            else:
                non_matching_size = 0

        if isinstance(dryspy_obj.nir, float):
            nothing_today = 1
        else:
            nothing_today = 0

        if non_matching_size or nothing_today:
            logging.warn("Skipping this image, not matching in size")
        else:
            logging.info("Detecting water...")
            dryspy_obj.detect_water(dryspy_obj.nir.copy())
            dryspy_obj.update_stats()  # Update the statistics
            # dryspy_obj.save_geotiff_results()

            if save_overview_pngs:
                logging.info("Saving png overviews...")
                dryspy_obj.save_classification_png(output_folder=output_folder)
            if save_tiff_results:
                logging.info("Saving tif results...")
                dryspy_obj.save_geotif_results(output_folder=output_folder)

    # Save statistics of the detected water
    dryspy_obj.temporal_stats()  # Do temporal statistics
    pdstats = pd.DataFrame(dryspy_obj.stats)
    pdstats.to_csv(os.path.join(output_folder, "stats.csv"))

    with open(
        os.path.join(output_folder, "config_of_run_" + os.path.basename(config_file)),
        "w",
    ) as configfile:
        dryspy_obj.config.write(configfile)


if __name__ == "__main__":
    run()
