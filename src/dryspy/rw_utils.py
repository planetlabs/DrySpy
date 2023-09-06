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

import rasterio
from rasterio.mask import mask
import rasterio.merge
from xml.dom import minidom
import glob, os
import fiona
import pandas as pd
import numpy as np
from google.cloud import storage
import datetime as dt


GS_SECRET_ACCESS_KEY = os.getenv("GS_SECRET_ACCESS_KEY")
GS_ACCESS_KEY_ID = os.getenv("GS_ACCESS_KEY_ID")
GOOGLE_APPLICATION_CREDENTIALS = "/Users/jschellekens/key.json"


def split_bucket_prefix(gcs_address):
    """
    Split a google cloud storage address into bucket name and prefix

    Parameters
    ----------
    gcs_address: string
        google cloud storage address starting with gs://

    Returns
    -------
    bucket_name: string
        name of the bucket
    prefix: string
        prefix in the bucket
    """
    url_parts = gcs_address.split("/")
    bucket_name = url_parts[2]
    prefix = "/".join(url_parts[3:])
    return bucket_name, prefix


def get_subscription_scene_paths(
    subscription_path: str, *, start_dt: dt.datetime, end_dt: dt.datetime
):
    """
    Iterates over scenes in a PlanetScope subscription.

    Parameters
    ----------
    subscription_path
        google storage path pointing to the planet-scope subscription
        e.g. gs://bp_test_api/8af4546e-76f3-4423-8fc8-b21f217660be
    start_dt
        iterate strips acquired from this datetime onward,
        microseconds are ignored
    end_dt
        iterate strips acquired until this datetime,
        microseconds are ignored


    Returns
    ------
    scene_metadata
        list of PSSceneMetadata

    """
    start_dt = dt.datetime(start_dt.year, start_dt.month, start_dt.day)
    storage_client = storage.Client()
    bucket_name, prefix = split_bucket_prefix(subscription_path)
    bucket = storage_client.get_bucket(bucket_name)
    start_offset = f"{prefix}/{start_dt:%Y%m%d_%H%M%S}"
    end_offset = f"{prefix}/{end_dt:%Y%m%d_%H%M%S}"
    print(prefix)
    blob_iterator = storage_client.list_blobs(
        bucket,
        prefix=prefix,
        fields="nextPageToken, items(name, id)",
        start_offset=start_offset,
        end_offset=end_offset,
    )
    PStiffiles = sorted(
        [
            blob
            for blob in blob_iterator
            if blob.name.endswith("AnalyticMS_SR_clip.tif")
        ],
        key=lambda blob: blob.name,
    )
    blob_iterator = storage_client.list_blobs(
        bucket,
        prefix=prefix,
        fields="nextPageToken, items(name, id)",
        start_offset=start_offset,
        end_offset=end_offset,
    )
    UDMtiffiles = sorted(
        [blob for blob in blob_iterator if blob.name.endswith("udm2_clip.tif")],
        key=lambda blob: blob.name,
    )
    blob_iterator = storage_client.list_blobs(
        bucket,
        prefix=prefix,
        fields="nextPageToken, items(name, id)",
        start_offset=start_offset,
        end_offset=end_offset,
    )
    PSxmlfiles = sorted(
        [
            blob
            for blob in blob_iterator
            if blob.name.endswith("AnalyticMS_metadata_clip.xml")
        ],
        key=lambda blob: blob.name,
    )

    toprocess = pd.DataFrame(
        {"Date": PStiffiles, "PS": PStiffiles, "UDM": UDMtiffiles, "XML": PSxmlfiles}
    )

    for line in range(0, len(toprocess)):
        toprocess.iloc[line]["PS"] = (
            "gs://"
            + toprocess.iloc[line]["PS"].bucket.name
            + "/"
            + toprocess.iloc[line]["PS"].name
        )
        toprocess.iloc[line]["UDM"] = (
            "gs://"
            + toprocess.iloc[line]["UDM"].bucket.name
            + "/"
            + toprocess.iloc[line]["UDM"].name
        )
        toprocess.iloc[line]["XML"] = (
            "gs://"
            + toprocess.iloc[line]["XML"].bucket.name
            + "/"
            + toprocess.iloc[line]["XML"].name
        )
        lst = toprocess.iloc[line]["PS"].split("/")
        print(lst)
        dtstr = lst[len(lst) - 2].split("_")[0]
        date = pd.Timestamp(dt.datetime.strptime(dtstr, "%Y%m%d").date())
        toprocess.iloc[line]["Date"] = date

    toprocess.set_index(["Date"])
    return toprocess


def construct_input_list_from_folder(
    input_folder, type="PS", startdate=None, enddate=None
):
    """generate a list of files to read from a folder, combine PS, udm and xml or use PF

    Parameters
    ----------
    input_folder : path
        the folder with a planetscope order
    type: string
        PS of PF (Planetscope or Planet Fusion)
    startdate: string
        date from which to start processing files (yyyy-mm-dd)
    enddate: string
        date to stop processing files (yyyy-mm-dd). if only startdate or enddate is given
        they are ignored, they are only used if both are given.

    Returns
    -------
    toprocess: pandas dataframe
        list of filenames to process
    """

    if type == "PS":
        PStiffiles = glob.glob(f"{input_folder}/*3B_AnalyticMS_SR_clip*.tif")
        UDMtiffiles = glob.glob(f"{input_folder}/*3B_udm2_clip*.tif")
        PSxmlfiles = glob.glob(f"{input_folder}/*AnalyticMS*.xml")
        PStiffiles.sort()
        UDMtiffiles.sort()
        PSxmlfiles.sort()

        dates = []
        for PS, UDM, XML in zip(PStiffiles, UDMtiffiles, PSxmlfiles):
            tmp = PS.split("_")
            thedate = os.path.basename(tmp[0]) + "_" + tmp[1]
            dates.append(thedate)

        toprocess = pd.DataFrame(
            {"Date": dates, "PS": PStiffiles, "UDM": UDMtiffiles, "XML": PSxmlfiles}
        )
        toprocess["Date"] = pd.to_datetime(toprocess["Date"], format="%Y%m%d_%H%M%S")
        toprocess = toprocess.set_index(["Date"])

        if (startdate is not None) and (enddate is not None):
            toprocess = toprocess.loc[startdate:enddate]
        return toprocess.sort_index()

    if type == "PF":
        PFtiffiles = glob.glob(f"{input_folder}/PF-SR/*.tif")
        QAtiffiles = glob.glob(f"{input_folder}/PF-QA/*.tif")

        dates = []
        for PF, QA in zip(PFtiffiles, QAtiffiles):
            thedate = os.path.splitext(os.path.basename(PF))[0]
            dates.append(thedate)

        toprocess = pd.DataFrame({"Date": dates, "PF": PFtiffiles, "QA": QAtiffiles})
        toprocess["Date"] = pd.to_datetime(toprocess["Date"], format="%Y-%m-%d")
        toprocess = toprocess.set_index(["Date"])
        if (startdate is not None) and (enddate is not None):
            toprocess = toprocess.loc[startdate:enddate]

        return toprocess.sort_index()


def construct_input_list_from_start_and_enddate(
    input_folder, type="PS", startdate=None, enddate=None
):
    date_list = pd.date_range(startdate, enddate, freq="D")

    if type == "PS":
        toprocess = get_subscription_scene_paths(
            input_folder,
            start_dt=dt.datetime.strptime(startdate, "%Y-%m-%d"),
            end_dt=dt.datetime.strptime(enddate, "%Y-%m-%d"),
        )
        toprocess = toprocess.set_index(["Date"])
        print(toprocess)

    if type == "PF":
        PFtiffiles = []
        QAtiffiles = []

        for date in date_list:
            PFtiffiles.append(input_folder + "/PF-SR/" + str(date.date()) + ".tif")
            QAtiffiles.append(input_folder + "/PF-QA/" + str(date.date()) + ".tif")

        toprocess = pd.DataFrame(
            {"Date": date_list, "PF": PFtiffiles, "QA": QAtiffiles}
        )
        toprocess = toprocess.set_index(["Date"])

    if type == "S2":
        S2tiffiles = []

        for date in date_list:
            S2tiffiles.append(
                input_folder
                + "/"
                + str(date.date())
                + "-00:00_"
                + str(date.date())
                + "-23:59_Sentinel-2_L2A_BAND_(Raw).tiff"
            )

        toprocess = pd.DataFrame({"Date": date_list, "S2": S2tiffiles})
        toprocess = toprocess.set_index(["Date"])

    return toprocess.sort_index()


def read_and_correct_PF(pffile, qafile, clip_file=None):
    """Read PS tiff, correct with UDM info and coefficients from the xml

    Parameters
    ----------
    pffile : string
        filename to open for the PF data
    qafile :string
        filename to open for the QA data

    Returns
    -------
    red, green, blue, nir, invalid, src.meta.copy(): 2d array, 2d array, 2d array, 2d array, 2d array,
    rasterio meta data filtered and corrected content of PF data file
    """

    if clip_file is not None:
        with fiona.open(clip_file, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        # with rasterio.env.Env(
        #    GOOGLE_APPLICATION_CREDENTIALS=GOOGLE_APPLICATION_CREDENTIALS
        # ):
        with rasterio.open(pffile) as src:
            pf, pf_trans = mask(src, shapes, crop=True)
        with rasterio.open(qafile) as qasrc:
            qa, qa_trans = mask(qasrc, shapes, crop=True)
            out_meta = src.meta.copy()
            out_meta_qa = qasrc.meta.copy()
    else:
        # with rasterio.env.Env(
        #    GOOGLE_APPLICATION_CREDENTIALS=GOOGLE_APPLICATION_CREDENTIALS
        # ):
        src = rasterio.open(pffile)
        qasrc = rasterio.open(qafile)
        pf = src.read()
        qa = qasrc.read()
        out_meta = src.meta.copy()
        out_meta_qa = qasrc.meta.copy()

    if clip_file is not None:
        out_meta.update(
            {
                "driver": "Gtiff",
                "height": pf.shape[1],  # height starts with shape[1]
                "width": pf.shape[2],  # width starts with shape[2]
                "transform": pf_trans,
            }
        )
        out_meta_qa.update(
            {
                "driver": "Gtiff",
                "height": qa.shape[1],  # height starts with shape[1]
                "width": qa.shape[2],  # width starts with shape[2]
                "transform": qa_trans,
            }
        )

    invalid = (pf[0, :, :] == 0.0).copy()

    # Multiply by corresponding coefficients
    blue = pf[0, :, :] * 0.0001
    green = pf[1, :, :] * 0.0001
    red = pf[2, :, :] * 0.0001
    nir = pf[3, :, :] * 0.0001

    # set the missing values to nan
    blue[pf[0, :, :] == 0] = np.nan
    green[pf[1, :, :] == 0] = np.nan
    red[pf[2, :, :] == 0] = np.nan
    nir[pf[3, :, :] == 0] = np.nan

    return red, green, blue, nir, invalid, out_meta, out_meta_qa, qa


def read_and_correct_PS(psfile, udmfile, xmlfile, clip_file=None, clonemap=None):
    """Read PS tiff, correct with UDM info and coefficients
       from the xml file.

    Parameters
    ----------
    psfile : string
        filename to open for the PS data
    udmfile :string
        filename to open for the UDM data
    xmlfile : string
        filename to open for the XML data

    Returns
    -------
    red, green, blue, nir, invalid, src.meta.copy(): 2d array, 2d array, 2d array, 2d array, 2d array, rasterio meta data
        filtered and corrected content of PS data file
    """

    if clip_file is not None:
        with fiona.open(clip_file, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        with rasterio.open(psfile) as src:
            ps, ps_trans = mask(src, shapes, crop=True, filled=True)
            out_meta = src.meta.copy()
        with rasterio.open(udmfile) as udmsrc:
            udm, udm_trans = mask(udmsrc, shapes, crop=True, filled=True)
            out_meta_udm = udmsrc.meta.copy()
    else:
        ps = src.read()
        udm = udmsrc.read()
        out_meta = src.meta.copy()
        out_meta_udm = udmsrc.meta.copy()

    if clip_file is not None:
        out_meta.update(
            {
                "driver": "Gtiff",
                "height": ps.shape[1],  # height starts with shape[1]
                "width": ps.shape[2],  # width starts with shape[2]
                "transform": ps_trans,
            }
        )
        out_meta_udm.update(
            {
                "driver": "Gtiff",
                "height": udm.shape[1],  # height starts with shape[1]
                "width": udm.shape[2],  # width starts with shape[2]
                "transform": udm_trans,
            }
        )

    invalid = udm[0, :, :] == 0

    if (
        xmlfile[0:5] == "gs://"
    ):  # Hack to get info from XML if stored on a google bucket
        storage_client = storage.Client()
        bucket_name, prefix = split_bucket_prefix(xmlfile)
        bucket = storage_client.get_bucket(bucket_name)
        # Create a blob object from the filepath
        blob = bucket.blob(prefix)
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        blob.download_to_filename(prefix)
        xmldoc = minidom.parse(prefix)
    else:
        xmldoc = minidom.parse(xmlfile)

    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ["1", "2", "3", "4"]:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[
                0
            ].firstChild.data
            coeffs[i] = float(value)

    # Multiply by corresponding coefficients
    green = ps[1, :, :] * coeffs[2]
    nir = ps[3, :, :] * coeffs[4]
    red = ps[0, :, :] * coeffs[1]
    blue = ps[2, :, :] * coeffs[3]

    # Set missing values to nan
    red[ps[0, :, :] == 0] = np.nan
    green[ps[1, :, :] == 0] = np.nan
    blue[ps[2, :, :] == 0] = np.nan
    nir[ps[3, :, :] == 0] = np.nan

    return red, green, blue, nir, invalid, out_meta, out_meta_udm, udm, src


def read_and_correct_S2(s2file, clip_file=None):
    """read s2 data exported from sentinel-hub (float32)as one file per band"""
    s2 = {}
    s2_trans = {}
    out_meta = {}
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

    if clip_file is not None:
        with fiona.open(clip_file, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        for band in bands:
            tmp = s2file.replace("BAND", band)
            if os.path.exists(tmp):
                with rasterio.open(tmp) as src:
                    s2[band], s2_trans[band] = mask(src, shapes, crop=True)
    else:
        for band in bands:
            tmp = s2file.replace("BAND", band)
            if os.path.exists(tmp):
                src = rasterio.open(tmp)
                s2[band] = src.read()
                out_meta[band] = src.meta.copy()

    if clip_file is not None:
        for band in bands:
            out_meta[band].update(
                {
                    "driver": "Gtiff",
                    "height": s2[band].shape[1],  # height starts with shape[1]
                    "width": s2[band].shape[2],  # width starts with shape[2]
                    "transform": s2_trans[band],
                }
            )

    if len(s2) >= 4:
        # Assume all bands have same invalid data
        invalid = (s2[band][0, :, :] == 0.0).copy()

        # Multiply by corresponding coefficients
        blue = s2["B02"][0, :, :].astype(float)
        green = s2["B03"][0, :, :].astype(float)
        red = s2["B04"][0, :, :].astype(float)
        nir = s2["B08"][0, :, :].astype(float)

        # set the missing values to nan
        blue[blue == 0] = np.nan
        green[green == 0] = np.nan
        red[red == 0] = np.nan
        nir[nir == 0] = np.nan
        return red, green, blue, nir, invalid, out_meta["B02"], None, None
    else:
        invalid = np.nan
        blue = np.nan
        red = np.nan
        green = np.nan
        nir = np.nan
        return red, green, blue, nir, invalid, None, None, None
