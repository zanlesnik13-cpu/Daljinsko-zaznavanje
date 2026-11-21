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
from typing import Any, Sequence, Union

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


StacItemOrGroup = Union[pystac.Item, Sequence[pystac.Item]]


def _normalize_item_group(items_or_item: StacItemOrGroup) -> list[pystac.Item]:
    """Normalize a single STAC Item or a sequence of same-day STAC Items into a non-empty list."""
    if isinstance(items_or_item, pystac.Item):
        return [items_or_item]
    item_list = [it for it in items_or_item if isinstance(it, pystac.Item)]
    if not item_list:
        raise ValueError("Expected at least one pystac.Item to load, got an empty sequence.")
    return item_list


def _extract_tile_label(item: pystac.Item, sensor: str) -> str:
    """Return a clean tile/path-row/orbit label for a STAC item."""
    if sensor == SENSOR_S2:
        tile = _extract_mgrs_tile(item)
        if not tile:
            return ""
        return tile if tile.startswith("T") else f"T{tile}"
    if sensor == SENSOR_LANDSAT:
        wrs_path = item.properties.get("landsat:wrs_path", "")
        wrs_row = item.properties.get("landsat:wrs_row", "")
        return f"P{wrs_path}/R{wrs_row}" if wrs_path and wrs_row else ""
    if sensor == SENSOR_S1:
        rel_orbit = item.properties.get("sat:relative_orbit", "")
        orbit_state = str(item.properties.get("sat:orbit_state", "")).upper()
        orbit_short = orbit_state[:3] if orbit_state else ""
        return f"Orb {rel_orbit} ({orbit_short})" if rel_orbit else orbit_short
    return _extract_mgrs_tile(item)


def _extract_platform_label(item: pystac.Item, sensor: str) -> str:
    """Return a clean platform label for a STAC item."""
    if sensor == SENSOR_LANDSAT:
        platform = str(item.properties.get("platform", ""))
        if _is_landsat7_slc_off(item):
            platform = f"{platform} (SLC-off)"
        return platform
    if sensor == SENSOR_S1:
        pols = item.properties.get("sar:polarizations") or []
        pol_str = "+".join(str(p) for p in pols)
        platform = str(item.properties.get("platform", "sentinel-1"))
        if pol_str:
            platform = f"{platform} ({pol_str})"
        return platform
    return str(item.properties.get("platform", ""))


def _merge_group_metadata(items: Sequence[pystac.Item], sensor: str) -> dict[str, Any]:
    """Summarize metadata across a group of same-day STAC items."""
    item_list = list(items)
    dts = [_extract_item_datetime_utc(it) for it in item_list]
    max_dt = max(dts)
    date_str = max_dt.strftime("%Y-%m-%d")

    tiles = sorted({lbl for it in item_list if (lbl := _extract_tile_label(it, sensor))})
    tile_str = " + ".join(tiles)

    platforms = sorted({lbl for it in item_list if (lbl := _extract_platform_label(it, sensor))})
    platform_str = " + ".join(platforms)

    if sensor == SENSOR_S1:
        cloud_mean = np.nan
        cloud_min = np.nan
    else:
        clouds: list[float] = []
        for it in item_list:
            raw_c = it.properties.get("eo:cloud_cover")
            if raw_c is not None:
                try:
                    val = float(raw_c)
                    if np.isfinite(val):
                        clouds.append(val)
                except (TypeError, ValueError):
                    pass
        cloud_mean = float(np.mean(clouds)) if clouds else np.nan
        cloud_min = float(np.min(clouds)) if clouds else np.nan

    primary_id = item_list[0].id
    scene_id = primary_id if len(item_list) == 1 else f"{primary_id} (+{len(item_list) - 1})"

    return {
        "id": scene_id,
        "datetime": max_dt,
        "date_str": date_str,
        "cloud_cover": cloud_mean,
        "cloud_cover_min": cloud_min,
        "mgrs_tile": tile_str,
        "platform": platform_str,
        "tile_count": len(item_list),
    }


def group_items_by_date(
    items: Sequence[pystac.Item],
    sensor: str,
    limit: int | None = None,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Group STAC items by acquisition date (``YYYY-MM-DD``) for seamless tile mosaicking.

    Returns ``(grouped_items, summary_df)`` ordered newest-first, where each entry
    in ``grouped_items`` is the list of all STAC items from that acquisition date
    covering the AOI (1-to-1 aligned with ``summary_df`` rows). Also attaches
    ``summary_df.attrs["items_by_date"] = {date_str: [item_1, item_2, ...]}``.
    """
    if not items:
        empty_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        empty_df.attrs["items_by_date"] = {}
        return [], empty_df

    buckets: dict[str, list[pystac.Item]] = {}
    for it in items:
        dt_utc = _extract_item_datetime_utc(it)
        date_str = dt_utc.strftime("%Y-%m-%d")
        buckets.setdefault(date_str, []).append(it)

    sorted_dates = sorted(buckets.keys(), reverse=True)
    if limit is not None and int(limit) > 0:
        sorted_dates = sorted_dates[: int(limit)]

    grouped_items: list[list[pystac.Item]] = []
    items_by_date: dict[str, list[pystac.Item]] = {}
    rows: list[dict[str, object]] = []

    for date_str in sorted_dates:
        day_items = buckets[date_str]
        if sensor == SENSOR_S1:
            day_items.sort(
                key=lambda it: (_extract_item_datetime_utc(it), it.id),
                reverse=True,
            )
        else:
            # Prefer clearer and more recently processed tiles first during mosaic fusion
            def _opt_sort_key(it: pystac.Item) -> tuple[float, float, str]:
                raw_c = it.properties.get("eo:cloud_cover")
                try:
                    c_val = float(raw_c) if raw_c is not None else float("inf")
                except (TypeError, ValueError):
                    c_val = float("inf")
                return (c_val, -_extract_item_datetime_utc(it).timestamp(), it.id)

            day_items.sort(key=_opt_sort_key)

        meta = _merge_group_metadata(day_items, sensor=sensor)
        grouped_items.append(day_items)
        items_by_date[date_str] = day_items
        rows.append(
            {
                "id": meta["id"],
                "datetime": pd.Timestamp(meta["datetime"]),
                "date_str": meta["date_str"],
                "cloud_cover": meta["cloud_cover"],
                "mgrs_tile": meta["mgrs_tile"],
                "platform": meta["platform"],
            }
        )

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_df.attrs["items_by_date"] = items_by_date
    return grouped_items, summary_df


def _collapse_time_dim(raw_ds: xr.Dataset, sensor: str) -> xr.Dataset:
    """Collapse a ``time`` dimension to a single 2D ``(y, x)`` mosaic slice."""
    if "time" not in raw_ds.dims:
        return raw_ds
    if raw_ds.sizes["time"] == 1:
        return raw_ds.isel(time=0)

    # Fallback mosaic if same-UTC-date items ever span two solar_day bins
    collapsed: dict[str, xr.DataArray] = {}
    spatial_coords = {k: v for k, v in raw_ds.coords.items() if "time" not in v.dims}
    for var_name in raw_ds.data_vars:
        da = raw_ds[var_name]
        arr_t = da.values  # (T, Y, X)
        out = arr_t[0].copy()
        nodata_val = da.attrs.get("nodata", None)

        def _is_valid(slice_arr: np.ndarray) -> np.ndarray:
            if sensor == SENSOR_S1:
                valid = np.isfinite(slice_arr) & (slice_arr > 0.0)
                if nodata_val is not None and np.isfinite(nodata_val):
                    valid &= slice_arr != nodata_val
                return valid
            if str(var_name) == "qa_pixel":
                return np.isfinite(slice_arr) & ((slice_arr.astype(np.uint16, copy=False) & 1) == 0)
            nd = 0 if nodata_val is None else nodata_val
            return np.isfinite(slice_arr) & (slice_arr != nd)

        filled = _is_valid(out)
        for t_idx in range(1, arr_t.shape[0]):
            if np.all(filled):
                break
            cand = arr_t[t_idx]
            take = (~filled) & _is_valid(cand)
            out[take] = cand[take]
            filled |= take

        collapsed[str(var_name)] = xr.DataArray(
            out,
            dims=da.dims[1:],
            coords={d: da.coords[d] for d in da.dims[1:] if d in da.coords},
            name=str(var_name),
            attrs=dict(da.attrs),
        )

    return xr.Dataset(data_vars=collapsed, coords=spatial_coords, attrs=dict(raw_ds.attrs))


# ---------------------------------------------------------------------------
# 1. SENTINEL-2 L2A
# ---------------------------------------------------------------------------
def search_sentinel2(
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    max_cloud_cover: float = 20,
    limit: int = 50,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Search Sentinel-2 L2A scenes on Planetary Computer, grouped by acquisition date."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    datetime_range = f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"
    raw_fetch_limit = max(int(limit) * 4, 150)

    catalog = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=[float(v) for v in bbox_4326],
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lte": float(max_cloud_cover)}},
        max_items=raw_fetch_limit,
    )

    raw_items = list(search.items())
    return group_items_by_date(raw_items, sensor=SENSOR_S2, limit=int(limit))


def load_sentinel2_scene(
    item: StacItemOrGroup,
    bbox_4326: Sequence[float],
    bands: Sequence[str],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load a clipped, radiometrically calibrated Sentinel-2 L2A scene (or same-day tile mosaic)."""
    band_list = [str(b).strip() for b in bands]
    if not band_list:
        raise ValueError("bands must contain at least one Sentinel-2 band name.")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    item_list = _normalize_item_group(item)
    crs_str = _normalize_crs_str(target_epsg)
    signed_items = [planetary_computer.sign(it) for it in item_list]

    if all(b in CATEGORICAL_BANDS for b in band_list):
        resampling_cfg: str | dict[str, str] = "nearest"
    elif any(b in CATEGORICAL_BANDS for b in band_list):
        resampling_cfg = {b: "nearest" for b in band_list if b in CATEGORICAL_BANDS}
        resampling_cfg["*"] = "bilinear"
    else:
        resampling_cfg = "bilinear"

    _configure_odc_rio()

    # Pass nodata=0 explicitly so overlapping same-day MGRS tiles fuse cleanly without
    # zero-padded swath edges overwriting valid reflectance or SCL pixels.
    raw_ds = odc.stac.load(
        signed_items,
        bands=band_list,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling=resampling_cfg,
        groupby="solar_day",
        nodata=0,
    )
    raw_ds = _collapse_time_dim(raw_ds, sensor=SENSOR_S2)

    primary_item = signed_items[0]
    apply_offset = _requires_pb04_offset(primary_item)
    calibrated_vars: dict[str, xr.DataArray] = {}

    for band in band_list:
        if band not in raw_ds.data_vars:
            raise KeyError(
                f"Requested band {band!r} was not loaded from STAC item(s) {[it.id for it in signed_items]!r}."
            )

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

    meta = _merge_group_metadata(signed_items, sensor=SENSOR_S2)
    ds = xr.Dataset(
        data_vars=calibrated_vars,
        coords=raw_ds.coords,
        attrs={
            "sensor": SENSOR_S2,
            "scene_id": meta["id"],
            "datetime": meta["datetime"].isoformat(),
            "cloud_cover": meta["cloud_cover"],
            "mgrs_tile": meta["mgrs_tile"],
            "tile_count": meta["tile_count"],
            "platform": meta["platform"],
            "processing_baseline": primary_item.properties.get("s2:processing_baseline"),
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
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Search Landsat 4/5/7/8/9 Collection 2 Level-2 scenes, grouped by acquisition date."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    datetime_range = f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"
    catalog = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)

    # Fetch extra items to avoid splitting same-day adjacent paths/rows or starving after L7 filtering
    fetch_limit = max(int(limit) * 6, 200) if exclude_l7_slc_off else max(int(limit) * 4, 150)
    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        bbox=[float(v) for v in bbox_4326],
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lte": float(max_cloud_cover)}},
        max_items=fetch_limit,
    )

    raw_items = list(search.items())
    if exclude_l7_slc_off:
        raw_items = [it for it in raw_items if not _is_landsat7_slc_off(it)]

    return group_items_by_date(raw_items, sensor=SENSOR_LANDSAT, limit=int(limit))


def load_landsat_scene(
    item: StacItemOrGroup,
    bbox_4326: Sequence[float],
    bands: Sequence[str],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 30,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load a clipped Landsat C2 L2 scene (or same-day mosaic) with harmonized band names.

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

    item_list = _normalize_item_group(item)
    crs_str = _normalize_crs_str(target_epsg)
    signed_items = [planetary_computer.sign(it) for it in item_list]

    if all(a == "qa_pixel" for a in asset_list):
        resampling_cfg: str | dict[str, str] = "nearest"
    elif "qa_pixel" in asset_list:
        resampling_cfg = {"qa_pixel": "nearest", "*": "bilinear"}
    else:
        resampling_cfg = "bilinear"

    _configure_odc_rio()

    raw_ds = odc.stac.load(
        signed_items,
        bands=asset_list,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling=resampling_cfg,
        groupby="solar_day",
    )
    raw_ds = _collapse_time_dim(raw_ds, sensor=SENSOR_LANDSAT)

    calibrated_vars: dict[str, xr.DataArray] = {}
    for asset_name in asset_list:
        if asset_name not in raw_ds.data_vars:
            raise KeyError(
                f"Asset {asset_name!r} was not loaded from Landsat item(s) {[it.id for it in signed_items]!r}."
            )

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

    meta = _merge_group_metadata(signed_items, sensor=SENSOR_LANDSAT)
    ds = xr.Dataset(
        data_vars=calibrated_vars,
        coords=raw_ds.coords,
        attrs={
            "sensor": SENSOR_LANDSAT,
            "scene_id": meta["id"],
            "datetime": meta["datetime"].isoformat(),
            "cloud_cover": meta["cloud_cover"],
            "mgrs_tile": meta["mgrs_tile"],
            "tile_count": meta["tile_count"],
            "platform": meta["platform"],
            "slc_off": any(_is_landsat7_slc_off(it) for it in signed_items),
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
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Search Sentinel-1 SAR RTC scenes on Planetary Computer, grouped by acquisition date."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    datetime_range = f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"
    raw_fetch_limit = max(int(limit) * 4, 150)

    catalog = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[SENTINEL1_COLLECTION],
        bbox=[float(v) for v in bbox_4326],
        datetime=datetime_range,
        max_items=raw_fetch_limit,
    )

    raw_items = list(search.items())
    return group_items_by_date(raw_items, sensor=SENSOR_S1, limit=int(limit))


def load_sentinel1_scene(
    item: StacItemOrGroup,
    bbox_4326: Sequence[float],
    polarizations: Sequence[str] = ("vv", "vh"),
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load a clipped Sentinel-1 RTC scene (or same-day granule mosaic) in decibels (dB).

    Converts linear intensity ``val`` to:
    ``sigma0_dB = 10 * log10(max(val, 1e-6))`` (``float32``), keeping non-finite
    and non-positive nodata values as ``NaN``. Variables are returned as uppercase
    ``VV`` and/or ``VH``.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    item_list = _normalize_item_group(item)
    req_pols = [str(p).strip().lower() for p in polarizations if str(p).strip()]
    available_pols = [p for p in req_pols if any(p in it.assets for it in item_list)]
    if not available_pols:
        all_assets = sorted({k for it in item_list for k in it.assets.keys()})
        raise KeyError(
            f"Requested polarizations {req_pols} not available in Sentinel-1 item(s) "
            f"{[it.id for it in item_list]!r}. Available assets: {all_assets}."
        )

    usable_items = [it for it in item_list if all(p in it.assets for p in available_pols)]
    if not usable_items:
        usable_items = [it for it in item_list if any(p in it.assets for p in available_pols)]

    crs_str = _normalize_crs_str(target_epsg)
    signed_items = [planetary_computer.sign(it) for it in usable_items]
    _configure_odc_rio()

    raw_ds = odc.stac.load(
        signed_items,
        bands=available_pols,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling="bilinear",
        groupby="solar_day",
    )
    raw_ds = _collapse_time_dim(raw_ds, sensor=SENSOR_S1)

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

    meta = _merge_group_metadata(signed_items, sensor=SENSOR_S1)
    ds = xr.Dataset(
        data_vars=calibrated_vars,
        coords=raw_ds.coords,
        attrs={
            "sensor": SENSOR_S1,
            "scene_id": meta["id"],
            "datetime": meta["datetime"].isoformat(),
            "cloud_cover": np.nan,
            "mgrs_tile": meta["mgrs_tile"],
            "tile_count": meta["tile_count"],
            "platform": meta["platform"],
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
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Dispatch date-grouped STAC search to Sentinel-2 L2A, Landsat C2 L2, or Sentinel-1 RTC."""
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
    item: StacItemOrGroup,
    bbox_4326: Sequence[float],
    bands: Sequence[str],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Dispatch clipped COG loading and same-day tile mosaicking to the selected sensor loader."""
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
