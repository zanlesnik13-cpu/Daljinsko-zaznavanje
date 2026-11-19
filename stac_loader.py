"""Cloud-native multi-sensor STAC search and calibrated COG loader.

Supports:
- Sentinel-2 L2A (`sentinel-2-l2a`, 10 m, 2015–present) with ESA PB >= 04.00
  radiometric offset calibration and SCL scene classification.
- Landsat Collection 2 Level-2 (`landsat-c2-l2`, 30 m, 1984–present) with
  harmonized band mapping (`B02/B03/B04/B08/B11/B12/QA_PIXEL`), official USGS
  surface reflectance scaling (`DN * 0.0000275 - 0.2`), and Landsat 7 SLC-off
  flagging/filtering.
- Sentinel-1 SAR RTC (`sentinel-1-rtc`, 10 m / 20 m, all-weather) with linear
  backscatter intensity converted to decibels (`10 * log10(max(val, 1e-6))`).

All scenes stream directly from Microsoft Planetary Computer COGs onto exact
EPSG:3794 (D96/TM) or EPSG:32633 (UTM 33N) grids with optional polygon clipping.
"""

from __future__ import annotations

import gc
import os
import re
from datetime import date, datetime, timezone
from typing import Any, Sequence

# Ensure GDAL/rasterio honors custom system/proxy CA bundles before library init
_ca_bundle = os.environ.get("CURL_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
if _ca_bundle and os.path.exists(_ca_bundle):
    os.environ.setdefault("GDAL_CURL_CA_BUNDLE", _ca_bundle)

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac
import rioxarray  # noqa: F401 - registers the .rio accessor on xarray objects
import xarray as xr
from pystac_client import Client

from geodezija_crs import clip_to_aoi_polygon

STAC_API_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Supported sensor collection identifiers
SENSOR_S2: str = "sentinel-2-l2a"
SENSOR_LANDSAT: str = "landsat-c2-l2"
SENSOR_S1: str = "sentinel-1-rtc"

SENTINEL2_COLLECTION: str = SENSOR_S2
LANDSAT_COLLECTION: str = SENSOR_LANDSAT
SENTINEL1_COLLECTION: str = SENSOR_S1

# ESA Sentinel-2 Processing Baseline 04.00 activation date (added +1000 BOA offset)
PB04_START_UTC: datetime = datetime(2022, 1, 25, 0, 0, 0, tzinfo=timezone.utc)
REFLECTANCE_SCALE: float = 10000.0
PB04_RADIOMETRIC_OFFSET: float = 1000.0

# USGS Landsat Collection 2 Level-2 optical reflectance calibration constants
LANDSAT_SR_SCALE: float = 0.0000275
LANDSAT_SR_OFFSET: float = -0.2
LANDSAT7_SLC_OFF_UTC: datetime = datetime(2003, 5, 31, 0, 0, 0, tzinfo=timezone.utc)

# Sentinel-1 SAR dB floor
SAR_MIN_LINEAR: float = 1e-6

OPTICAL_BANDS: frozenset[str] = frozenset(
    {
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
    }
)
CATEGORICAL_BANDS: frozenset[str] = frozenset({"SCL", "QA_PIXEL", "qa_pixel"})

# Harmonized mapping between Sentinel-2 logical band names and Planetary Computer Landsat assets
LOGICAL_TO_LANDSAT_ASSET: dict[str, str] = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir08",
    "B11": "swir16",
    "B12": "swir22",
    "QA_PIXEL": "qa_pixel",
    "SCL": "qa_pixel",
    # Direct PC names also accepted
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir08": "nir08",
    "swir16": "swir16",
    "swir22": "swir22",
    "qa_pixel": "qa_pixel",
}
LANDSAT_ASSET_TO_LOGICAL: dict[str, str] = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir08": "B08",
    "swir16": "B11",
    "swir22": "B12",
    "qa_pixel": "QA_PIXEL",
}

SUMMARY_COLUMNS: list[str] = [
    "id",
    "datetime",
    "date_str",
    "cloud_cover",
    "mgrs_tile",
    "platform",
]

_MGRS_REGEX = re.compile(r"_T([0-9]{2}[A-Z]{3})_")


def _format_date_bound(value: str | date | datetime) -> str:
    """Format a date/datetime/string bound for STAC datetime range queries."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.strftime("%Y-%m-%dT%H:%M:%SZ")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def _normalize_crs_str(target_epsg: int | str) -> str:
    """Normalize EPSG integer or CRS string into a canonical CRS string."""
    if isinstance(target_epsg, int):
        return f"EPSG:{target_epsg}"
    text = str(target_epsg).strip()
    if text.isdigit():
        return f"EPSG:{text}"
    return text


def _configure_odc_rio() -> None:
    """Configure odc-stac GDAL/rasterio options for fast, RAM-safe COG streaming."""
    gdal_env_opts: dict[str, object] = {
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_CACHEMAX": 64,
    }
    ca_bundle = (
        os.environ.get("GDAL_CURL_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
    )
    if ca_bundle and os.path.exists(ca_bundle):
        os.environ.setdefault("GDAL_CURL_CA_BUNDLE", ca_bundle)
        gdal_env_opts["GDAL_CURL_CA_BUNDLE"] = ca_bundle

    odc.stac.configure_rio(cloud_defaults=True, **gdal_env_opts)


def _extract_item_datetime_utc(item: pystac.Item) -> datetime:
    """Return the item's acquisition datetime normalized to UTC."""
    dt = item.datetime
    if dt is None:
        raw = item.properties.get("datetime")
        if raw is None:
            raise ValueError(f"STAC item {item.id!r} has no acquisition datetime.")
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_mgrs_tile(item: pystac.Item) -> str:
    """Extract the 5-character MGRS tile identifier (e.g. '33TVM')."""
    tile = item.properties.get("s2:mgrs_tile")
    if tile:
        return str(tile)
    match = _MGRS_REGEX.search(item.id)
    if match:
        return match.group(1)
    return ""


def _parse_processing_baseline(item: pystac.Item) -> float | None:
    """Parse Sentinel-2 processing baseline (e.g. '05.11' -> 5.11) if present."""
    raw = item.properties.get("s2:processing_baseline")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _requires_pb04_offset(item: pystac.Item) -> bool:
    """Determine whether Sentinel-2 L2A DN values include the +1000 BOA offset."""
    baseline = _parse_processing_baseline(item)
    if baseline is not None:
        return baseline >= 4.0
    return _extract_item_datetime_utc(item) >= PB04_START_UTC


def _is_landsat7_slc_off(item: pystac.Item) -> bool:
    """Return True if a Landsat STAC item is Landsat-7 acquired after SLC failure (2003-05-31)."""
    platform = str(item.properties.get("platform", "")).lower()
    if "landsat-7" in platform or item.id.startswith("LE07"):
        return _extract_item_datetime_utc(item) >= LANDSAT7_SLC_OFF_UTC
    return False


# ---------------------------------------------------------------------------
# 1. SENTINEL-2 L2A
# ---------------------------------------------------------------------------
def search_sentinel2(
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    max_cloud_cover: float = 20,
    limit: int = 50,
) -> tuple[list[pystac.Item], pd.DataFrame]:
    """Search Sentinel-2 L2A scenes on Microsoft Planetary Computer STAC API."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    datetime_range = f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"

    catalog = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=[float(v) for v in bbox_4326],
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lte": float(max_cloud_cover)}},
        max_items=int(limit),
    )

    items = list(search.items())
    if not items:
        return [], pd.DataFrame(columns=SUMMARY_COLUMNS)

    items.sort(key=_extract_item_datetime_utc, reverse=True)

    rows: list[dict[str, object]] = []
    for item in items:
        dt_utc = _extract_item_datetime_utc(item)
        cloud_raw = item.properties.get("eo:cloud_cover")
        rows.append(
            {
                "id": item.id,
                "datetime": pd.Timestamp(dt_utc),
                "date_str": dt_utc.strftime("%Y-%m-%d"),
                "cloud_cover": float(cloud_raw) if cloud_raw is not None else np.nan,
                "mgrs_tile": _extract_mgrs_tile(item),
                "platform": str(item.properties.get("platform", "")),
            }
        )

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return items, summary_df


def load_sentinel2_scene(
    item: pystac.Item,
    bbox_4326: Sequence[float],
    bands: Sequence[str],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load a clipped, radiometrically calibrated Sentinel-2 L2A scene via COG streaming."""
    band_list = [str(b).strip() for b in bands]
    if not band_list:
        raise ValueError("bands must contain at least one Sentinel-2 band name.")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    crs_str = _normalize_crs_str(target_epsg)
    signed_item = planetary_computer.sign(item)

    if all(b in CATEGORICAL_BANDS for b in band_list):
        resampling_cfg: str | dict[str, str] = "nearest"
    elif any(b in CATEGORICAL_BANDS for b in band_list):
        resampling_cfg = {b: "nearest" for b in band_list if b in CATEGORICAL_BANDS}
        resampling_cfg["*"] = "bilinear"
    else:
        resampling_cfg = "bilinear"

    _configure_odc_rio()

    raw_ds = odc.stac.load(
        [signed_item],
        bands=band_list,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling=resampling_cfg,
    )

    if "time" in raw_ds.dims and raw_ds.sizes["time"] == 1:
        raw_ds = raw_ds.isel(time=0)

    apply_offset = _requires_pb04_offset(signed_item)
    calibrated_vars: dict[str, xr.DataArray] = {}

    for band in band_list:
        if band not in raw_ds.data_vars:
            raise KeyError(f"Requested band {band!r} was not loaded from STAC item {item.id!r}.")

        raw_da = raw_ds[band]
        if band in OPTICAL_BANDS:
            nodata_val = raw_da.attrs.get("nodata", 0)
            if nodata_val is None:
                nodata_val = 0

            raw_vals = raw_da.values
            valid_mask = np.isfinite(raw_vals) & (raw_vals != nodata_val)

            refl = raw_vals.astype(np.float32, copy=True)
            if apply_offset:
                refl -= np.float32(PB04_RADIOMETRIC_OFFSET)
            refl /= np.float32(REFLECTANCE_SCALE)
            np.clip(refl, np.float32(0.0), np.float32(1.0), out=refl)
            refl[~valid_mask] = np.nan

            cal_da = xr.DataArray(
                refl,
                dims=raw_da.dims,
                coords=raw_da.coords,
                name=band,
                attrs={
                    "units": "surface_reflectance",
                    "long_name": f"Sentinel-2 L2A {band} Surface Reflectance",
                    "radiometric_offset_applied": apply_offset,
                },
            )
            cal_da = cal_da.rio.write_crs(crs_str, inplace=False)
            cal_da = cal_da.rio.write_nodata(np.nan, inplace=False)
            calibrated_vars[band] = cal_da
            del raw_vals, valid_mask, refl
        elif band == "SCL":
            scl_raw = raw_da.values
            scl_vals = np.where(np.isfinite(scl_raw), scl_raw, 0).astype(np.uint8)
            scl_da = xr.DataArray(
                scl_vals,
                dims=raw_da.dims,
                coords=raw_da.coords,
                name="SCL",
                attrs={
                    "units": "class",
                    "long_name": "Sentinel-2 L2A Scene Classification Layer (SCL)",
                },
            )
            scl_da = scl_da.rio.write_crs(crs_str, inplace=False)
            scl_da = scl_da.rio.write_nodata(0, inplace=False)
            calibrated_vars["SCL"] = scl_da
        else:
            calibrated_vars[band] = raw_da.rio.write_crs(crs_str, inplace=False)

    dt_utc = _extract_item_datetime_utc(signed_item)
    ds = xr.Dataset(
        data_vars=calibrated_vars,
        coords=raw_ds.coords,
        attrs={
            "sensor": SENSOR_S2,
            "scene_id": signed_item.id,
            "datetime": dt_utc.isoformat(),
            "cloud_cover": signed_item.properties.get("eo:cloud_cover"),
            "mgrs_tile": _extract_mgrs_tile(signed_item),
            "platform": signed_item.properties.get("platform", ""),
            "processing_baseline": signed_item.properties.get("s2:processing_baseline"),
            "radiometric_offset_applied": apply_offset,
            "resolution_m": float(resolution),
            "crs": crs_str,
        },
    )
    ds = ds.rio.write_crs(crs_str, inplace=False)

    if aoi_polygon_4326 is not None:
        ds = clip_to_aoi_polygon(ds, aoi_polygon_4326, drop=False)

    del raw_ds, calibrated_vars
    gc.collect()
    return ds


# ---------------------------------------------------------------------------
# 2. LANDSAT COLLECTION 2 LEVEL-2 (1984–PRESENT)
# ---------------------------------------------------------------------------
def search_landsat(
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    max_cloud_cover: float = 20,
    limit: int = 50,
    exclude_l7_slc_off: bool = False,
) -> tuple[list[pystac.Item], pd.DataFrame]:
    """Search Landsat 4/5/7/8/9 Collection 2 Level-2 scenes on Planetary Computer."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    datetime_range = f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"
    catalog = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)

    # Fetch extra items if filtering out L7 SLC-off scenes so we still meet limit
    fetch_limit = int(limit * 2) if exclude_l7_slc_off else int(limit)
    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        bbox=[float(v) for v in bbox_4326],
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lte": float(max_cloud_cover)}},
        max_items=fetch_limit,
    )

    items = list(search.items())
    if exclude_l7_slc_off:
        items = [it for it in items if not _is_landsat7_slc_off(it)]
    items = items[: int(limit)]

    if not items:
        return [], pd.DataFrame(columns=SUMMARY_COLUMNS)

    items.sort(key=_extract_item_datetime_utc, reverse=True)

    rows: list[dict[str, object]] = []
    for item in items:
        dt_utc = _extract_item_datetime_utc(item)
        cloud_raw = item.properties.get("eo:cloud_cover")
        wrs_path = item.properties.get("landsat:wrs_path", "")
        wrs_row = item.properties.get("landsat:wrs_row", "")
        tile_label = f"P{wrs_path}/R{wrs_row}" if wrs_path and wrs_row else ""

        platform = str(item.properties.get("platform", ""))
        if _is_landsat7_slc_off(item):
            platform = f"{platform} (SLC-off)"

        rows.append(
            {
                "id": item.id,
                "datetime": pd.Timestamp(dt_utc),
                "date_str": dt_utc.strftime("%Y-%m-%d"),
                "cloud_cover": float(cloud_raw) if cloud_raw is not None else np.nan,
                "mgrs_tile": tile_label,
                "platform": platform,
            }
        )

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return items, summary_df


def load_landsat_scene(
    item: pystac.Item,
    bbox_4326: Sequence[float],
    bands: Sequence[str],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 30,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load a clipped Landsat C2 L2 scene mapped to harmonized Sentinel-2 logical band names.

    Applies official USGS Collection 2 Level-2 surface reflectance scaling:
    ``reflectance = DN * 0.0000275 - 0.2``, clipped to ``[0.0, 1.0]`` (``float32``),
    with fill pixels (``DN == 0``) masked to ``NaN``.
    Quality band ``qa_pixel`` is loaded with nearest-neighbor resampling as ``uint16``
    under the variable name ``QA_PIXEL``.
    """
    if not bands:
        raise ValueError("bands must contain at least one band name.")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    # Translate requested logical bands (e.g., B04, B08, SCL/QA_PIXEL) to PC Landsat assets
    asset_list: list[str] = []
    for b in bands:
        key = str(b).strip()
        pc_asset = LOGICAL_TO_LANDSAT_ASSET.get(key)
        if pc_asset is None:
            raise KeyError(
                f"Band {key!r} is not mapped for Landsat C2 L2. "
                f"Supported bands: B02, B03, B04, B08, B11, B12, QA_PIXEL."
            )
        if pc_asset not in asset_list:
            asset_list.append(pc_asset)

    crs_str = _normalize_crs_str(target_epsg)
    signed_item = planetary_computer.sign(item)

    if all(a == "qa_pixel" for a in asset_list):
        resampling_cfg: str | dict[str, str] = "nearest"
    elif "qa_pixel" in asset_list:
        resampling_cfg = {"qa_pixel": "nearest", "*": "bilinear"}
    else:
        resampling_cfg = "bilinear"

    _configure_odc_rio()

    raw_ds = odc.stac.load(
        [signed_item],
        bands=asset_list,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling=resampling_cfg,
    )

    if "time" in raw_ds.dims and raw_ds.sizes["time"] == 1:
        raw_ds = raw_ds.isel(time=0)

    calibrated_vars: dict[str, xr.DataArray] = {}
    for asset_name in asset_list:
        if asset_name not in raw_ds.data_vars:
            raise KeyError(f"Asset {asset_name!r} was not loaded from Landsat item {item.id!r}.")

        logical_name = LANDSAT_ASSET_TO_LOGICAL.get(asset_name, asset_name)
        raw_da = raw_ds[asset_name]

        if asset_name == "qa_pixel":
            qa_raw = raw_da.values
            qa_vals = np.where(np.isfinite(qa_raw), qa_raw, 1).astype(np.uint16)
            qa_da = xr.DataArray(
                qa_vals,
                dims=raw_da.dims,
                coords=raw_da.coords,
                name="QA_PIXEL",
                attrs={
                    "units": "bitmask",
                    "long_name": "Landsat Collection 2 L2 Pixel Quality Assessment (QA_PIXEL)",
                },
            )
            qa_da = qa_da.rio.write_crs(crs_str, inplace=False)
            qa_da = qa_da.rio.write_nodata(0, inplace=False)
            calibrated_vars["QA_PIXEL"] = qa_da
        else:
            raw_vals = raw_da.values
            valid_mask = np.isfinite(raw_vals) & (raw_vals != 0)

            refl = raw_vals.astype(np.float32, copy=True)
            refl *= np.float32(LANDSAT_SR_SCALE)
            refl += np.float32(LANDSAT_SR_OFFSET)
            np.clip(refl, np.float32(0.0), np.float32(1.0), out=refl)
            refl[~valid_mask] = np.nan

            cal_da = xr.DataArray(
                refl,
                dims=raw_da.dims,
                coords=raw_da.coords,
                name=logical_name,
                attrs={
                    "units": "surface_reflectance",
                    "long_name": f"Landsat C2 L2 {logical_name} ({asset_name}) Surface Reflectance",
                    "source_asset": asset_name,
                },
            )
            cal_da = cal_da.rio.write_crs(crs_str, inplace=False)
            cal_da = cal_da.rio.write_nodata(np.nan, inplace=False)
            calibrated_vars[logical_name] = cal_da
            del raw_vals, valid_mask, refl

    dt_utc = _extract_item_datetime_utc(signed_item)
    wrs_path = signed_item.properties.get("landsat:wrs_path", "")
    wrs_row = signed_item.properties.get("landsat:wrs_row", "")
    tile_label = f"P{wrs_path}/R{wrs_row}" if wrs_path and wrs_row else ""

    ds = xr.Dataset(
        data_vars=calibrated_vars,
        coords=raw_ds.coords,
        attrs={
            "sensor": SENSOR_LANDSAT,
            "scene_id": signed_item.id,
            "datetime": dt_utc.isoformat(),
            "cloud_cover": signed_item.properties.get("eo:cloud_cover"),
            "mgrs_tile": tile_label,
            "platform": signed_item.properties.get("platform", ""),
            "slc_off": _is_landsat7_slc_off(signed_item),
            "processing_baseline": "C2-L2",
            "radiometric_offset_applied": True,
            "resolution_m": float(resolution),
            "crs": crs_str,
        },
    )
    ds = ds.rio.write_crs(crs_str, inplace=False)

    if aoi_polygon_4326 is not None:
        ds = clip_to_aoi_polygon(ds, aoi_polygon_4326, drop=False)

    del raw_ds, calibrated_vars
    gc.collect()
    return ds


# ---------------------------------------------------------------------------
# 3. SENTINEL-1 SAR RTC (ALL-WEATHER RADAR)
# ---------------------------------------------------------------------------
def search_sentinel1(
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    limit: int = 50,
) -> tuple[list[pystac.Item], pd.DataFrame]:
    """Search Sentinel-1 SAR RTC scenes on Microsoft Planetary Computer (all-weather)."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    datetime_range = f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"
    catalog = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[SENTINEL1_COLLECTION],
        bbox=[float(v) for v in bbox_4326],
        datetime=datetime_range,
        max_items=int(limit),
    )

    items = list(search.items())
    if not items:
        return [], pd.DataFrame(columns=SUMMARY_COLUMNS)

    items.sort(key=_extract_item_datetime_utc, reverse=True)

    rows: list[dict[str, object]] = []
    for item in items:
        dt_utc = _extract_item_datetime_utc(item)
        rel_orbit = item.properties.get("sat:relative_orbit", "")
        orbit_state = str(item.properties.get("sat:orbit_state", "")).upper()
        orbit_short = orbit_state[:3] if orbit_state else ""
        tile_label = f"Orb {rel_orbit} ({orbit_short})" if rel_orbit else orbit_short

        pols = item.properties.get("sar:polarizations") or []
        pol_str = "+".join(str(p) for p in pols)
        platform = str(item.properties.get("platform", "sentinel-1"))
        if pol_str:
            platform = f"{platform} ({pol_str})"

        rows.append(
            {
                "id": item.id,
                "datetime": pd.Timestamp(dt_utc),
                "date_str": dt_utc.strftime("%Y-%m-%d"),
                "cloud_cover": np.nan,
                "mgrs_tile": tile_label,
                "platform": platform,
            }
        )

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return items, summary_df


def load_sentinel1_scene(
    item: pystac.Item,
    bbox_4326: Sequence[float],
    polarizations: Sequence[str] = ("vv", "vh"),
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load a clipped Sentinel-1 RTC scene and convert linear backscatter to decibels (dB).

    Converts linear intensity ``val`` to:
    ``sigma0_dB = 10 * log10(max(val, 1e-6))`` (``float32``), keeping non-finite
    and non-positive nodata values as ``NaN``. Variables are returned as uppercase
    ``VV`` and/or ``VH``.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    req_pols = [str(p).strip().lower() for p in polarizations if str(p).strip()]
    available_pols = [p for p in req_pols if p in item.assets]
    if not available_pols:
        raise KeyError(
            f"Requested polarizations {req_pols} not available in Sentinel-1 item {item.id!r}. "
            f"Available assets: {sorted(item.assets.keys())}."
        )

    crs_str = _normalize_crs_str(target_epsg)
    signed_item = planetary_computer.sign(item)
    _configure_odc_rio()

    raw_ds = odc.stac.load(
        [signed_item],
        bands=available_pols,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling="bilinear",
    )

    if "time" in raw_ds.dims and raw_ds.sizes["time"] == 1:
        raw_ds = raw_ds.isel(time=0)

    calibrated_vars: dict[str, xr.DataArray] = {}
    for pol in available_pols:
        raw_da = raw_ds[pol]
        raw_vals = raw_da.values.astype(np.float32, copy=False)
        nodata_val = raw_da.attrs.get("nodata", None)

        valid_mask = np.isfinite(raw_vals) & (raw_vals > 0.0)
        if nodata_val is not None and np.isfinite(nodata_val):
            valid_mask &= raw_vals != nodata_val

        db_vals = np.full(raw_vals.shape, np.nan, dtype=np.float32)
        if np.any(valid_mask):
            safe_linear = np.maximum(raw_vals[valid_mask], np.float32(SAR_MIN_LINEAR))
            db_vals[valid_mask] = np.float32(10.0) * np.log10(safe_linear)

        pol_upper = pol.upper()
        pol_da = xr.DataArray(
            db_vals,
            dims=raw_da.dims,
            coords=raw_da.coords,
            name=pol_upper,
            attrs={
                "units": "dB",
                "long_name": f"Sentinel-1 RTC {pol_upper} Backscatter [dB]",
            },
        )
        pol_da = pol_da.rio.write_crs(crs_str, inplace=False)
        pol_da = pol_da.rio.write_nodata(np.nan, inplace=False)
        calibrated_vars[pol_upper] = pol_da
        del raw_vals, valid_mask, db_vals

    dt_utc = _extract_item_datetime_utc(signed_item)
    rel_orbit = signed_item.properties.get("sat:relative_orbit", "")
    orbit_state = str(signed_item.properties.get("sat:orbit_state", "")).upper()
    tile_label = f"Orb {rel_orbit} ({orbit_state[:3]})" if rel_orbit else orbit_state[:3]

    ds = xr.Dataset(
        data_vars=calibrated_vars,
        coords=raw_ds.coords,
        attrs={
            "sensor": SENSOR_S1,
            "scene_id": signed_item.id,
            "datetime": dt_utc.isoformat(),
            "cloud_cover": np.nan,
            "mgrs_tile": tile_label,
            "platform": signed_item.properties.get("platform", "sentinel-1"),
            "processing_baseline": "RTC",
            "radiometric_offset_applied": False,
            "resolution_m": float(resolution),
            "crs": crs_str,
        },
    )
    ds = ds.rio.write_crs(crs_str, inplace=False)

    if aoi_polygon_4326 is not None:
        ds = clip_to_aoi_polygon(ds, aoi_polygon_4326, drop=False)

    del raw_ds, calibrated_vars
    gc.collect()
    return ds


# ---------------------------------------------------------------------------
# 4. UNIFIED MULTI-SENSOR DISPATCHERS
# ---------------------------------------------------------------------------
def search_scenes(
    sensor: str,
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    max_cloud_cover: float = 20,
    limit: int = 50,
    exclude_l7_slc_off: bool = False,
) -> tuple[list[pystac.Item], pd.DataFrame]:
    """Dispatch STAC search to Sentinel-2 L2A, Landsat C2 L2, or Sentinel-1 RTC."""
    if sensor == SENSOR_S2:
        return search_sentinel2(
            bbox_4326=bbox_4326,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
        )
    if sensor == SENSOR_LANDSAT:
        return search_landsat(
            bbox_4326=bbox_4326,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
            exclude_l7_slc_off=exclude_l7_slc_off,
        )
    if sensor == SENSOR_S1:
        return search_sentinel1(
            bbox_4326=bbox_4326,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    raise ValueError(f"Unsupported sensor identifier: {sensor!r}.")


def load_scene(
    sensor: str,
    item: pystac.Item,
    bbox_4326: Sequence[float],
    bands: Sequence[str],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Dispatch clipped COG loading and calibration to the selected sensor loader."""
    if sensor == SENSOR_S2:
        return load_sentinel2_scene(
            item=item,
            bbox_4326=bbox_4326,
            bands=bands,
            target_epsg=target_epsg,
            resolution=resolution,
            aoi_polygon_4326=aoi_polygon_4326,
        )
    if sensor == SENSOR_LANDSAT:
        return load_landsat_scene(
            item=item,
            bbox_4326=bbox_4326,
            bands=bands,
            target_epsg=target_epsg,
            resolution=resolution,
            aoi_polygon_4326=aoi_polygon_4326,
        )
    if sensor == SENSOR_S1:
        return load_sentinel1_scene(
            item=item,
            bbox_4326=bbox_4326,
            polarizations=bands,
            target_epsg=target_epsg,
            resolution=resolution,
            aoi_polygon_4326=aoi_polygon_4326,
        )
    raise ValueError(f"Unsupported sensor identifier: {sensor!r}.")
