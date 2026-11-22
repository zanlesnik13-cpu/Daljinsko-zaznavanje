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
from datetime import date, datetime, timedelta, timezone
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
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union as shapely_unary_union

from geodezija_crs import clip_to_aoi_polygon

STAC_API_URL: str = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Supported sensor collection identifiers
SENSOR_S2: str = "sentinel-2-l2a"
SENSOR_LANDSAT: str = "landsat-c2-l2"
SENSOR_S1: str = "sentinel-1-rtc"

SENTINEL2_COLLECTION: str = SENSOR_S2
LANDSAT_COLLECTION: str = SENSOR_LANDSAT
SENTINEL1_COLLECTION: str = SENSOR_S1

# Historical archive bounds, operational Landsat missions & swath coverage defaults
HISTORICAL_MIN_DATE: date = date(1984, 1, 1)
DEFAULT_MIN_AOI_COVERAGE_PCT: float = 90.0
LANDSAT_OPERATIONAL_PLATFORMS: list[str] = [
    "landsat-4",
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "landsat-9",
]

# Temporal sampling interval identifiers
INTERVAL_ALL: str = "all"
INTERVAL_14D: str = "14d"
INTERVAL_MONTHLY: str = "monthly"
INTERVAL_SEASONAL: str = "seasonal"
INTERVAL_YEARLY: str = "yearly"
TEMPORAL_INTERVAL_KEYS: tuple[str, ...] = (
    INTERVAL_ALL,
    INTERVAL_14D,
    INTERVAL_MONTHLY,
    INTERVAL_SEASONAL,
    INTERVAL_YEARLY,
)

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
    "coverage_pct",
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


def _coerce_date(value: str | date | datetime) -> date:
    """Convert a date, datetime, or ISO string into a calendar ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    return date.fromisoformat(raw[:10])


def _build_stac_date_windows(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    chunk_years: int = 5,
) -> list[str]:
    """Split spans > 3 years into multi-year STAC windows so old decades are never truncated."""
    d_start = _coerce_date(start_date)
    d_end = _coerce_date(end_date)
    if d_end < d_start:
        d_start, d_end = d_end, d_start

    span_days = (d_end - d_start).days
    if span_days <= 3 * 365:
        return [f"{_format_date_bound(start_date)}/{_format_date_bound(end_date)}"]

    windows: list[str] = []
    cur_start = d_start
    while cur_start <= d_end:
        try:
            next_year_start = date(cur_start.year + chunk_years, cur_start.month, cur_start.day)
        except ValueError:
            next_year_start = date(cur_start.year + chunk_years, cur_start.month, 28)
        cur_end = min(d_end, next_year_start - timedelta(days=1))
        windows.append(f"{cur_start.isoformat()}/{cur_end.isoformat()}")
        cur_start = cur_end + timedelta(days=1)

    return windows


def _fetch_stac_items_full(
    *,
    collection: str,
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    query: dict[str, Any] | None = None,
    max_items_per_window: int = 1000,
    page_limit: int = 250,
) -> list[pystac.Item]:
    """Fetch STAC items across the entire requested historical span without cutting off older years.

    Uses a single shared unsigned STAC API client iterating cleanly across 5-year windows
    (avoiding SAS signing overhead and API burst rate-limits at search time; items are
    freshly signed inside ``load_*_scene`` right before COG streaming).
    """
    from pystac_client.stac_api_io import StacApiIO

    bbox_floats = [float(v) for v in bbox_4326]
    windows = _build_stac_date_windows(start_date, end_date, chunk_years=5)
    per_window_cap = max_items_per_window if len(windows) == 1 else min(max_items_per_window, 500)

    catalog = Client.open(STAC_API_URL, stac_io=StacApiIO(timeout=25))
    raw_items: list[pystac.Item] = []

    for dt_range in windows:
        kwargs: dict[str, Any] = {
            "collections": [collection],
            "bbox": bbox_floats,
            "datetime": dt_range,
            "limit": int(page_limit),
            "max_items": int(per_window_cap),
        }
        if query:
            kwargs["query"] = query
        search = catalog.search(**kwargs)
        raw_items.extend(search.items())

    # Deduplicate by STAC item ID in case window boundaries overlap
    seen_ids: set[str] = set()
    unique_items: list[pystac.Item] = []
    for it in raw_items:
        if it.id not in seen_ids:
            seen_ids.add(it.id)
            unique_items.append(it)
    return unique_items


def _build_aoi_geom_4326(
    bbox_4326: Sequence[float] | None,
    aoi_polygon_4326: dict[str, Any] | None = None,
):
    """Build a valid Shapely polygon in EPSG:4326 for fast STAC footprint coverage checks."""
    if aoi_polygon_4326 is not None:
        try:
            geom = shapely_shape(aoi_polygon_4326)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_empty and geom.area > 0:
                return geom
        except Exception:
            pass
    if bbox_4326 is not None and len(bbox_4326) == 4:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
        if min_lon < max_lon and min_lat < max_lat:
            return shapely_box(min_lon, min_lat, max_lon, max_lat)
    return None


def _compute_group_aoi_coverage_pct(
    day_items: Sequence[pystac.Item],
    aoi_geom: Any,
) -> float:
    """Compute geometric AOI coverage [%] from the union of same-day STAC item footprints."""
    if aoi_geom is None or aoi_geom.is_empty or aoi_geom.area <= 0:
        return 100.0

    geoms = []
    for it in day_items:
        g = None
        if getattr(it, "geometry", None):
            try:
                g = shapely_shape(it.geometry)
                if not g.is_valid:
                    g = g.buffer(0)
            except Exception:
                g = None
        if (g is None or g.is_empty) and getattr(it, "bbox", None) and len(it.bbox) >= 4:
            try:
                g = shapely_box(float(it.bbox[0]), float(it.bbox[1]), float(it.bbox[2]), float(it.bbox[3]))
            except Exception:
                g = None
        if g is not None and not g.is_empty:
            geoms.append(g)

    if not geoms:
        return 0.0

    try:
        union_geom = geoms[0] if len(geoms) == 1 else shapely_unary_union(geoms)
        if not union_geom.is_valid:
            union_geom = union_geom.buffer(0)
        inter_area = float(union_geom.intersection(aoi_geom).area)
        cov = (inter_area / float(aoi_geom.area)) * 100.0
        return round(float(np.clip(cov, 0.0, 100.0)), 1)
    except Exception:
        return 100.0


class _NoDeepCopyDict(dict):
    """Dict subclass for ``DataFrame.attrs`` that prevents Pandas from deep-copying ``pystac.Item`` trees."""

    def __deepcopy__(self, memo: dict[int, Any]) -> "_NoDeepCopyDict":
        return self


class _ShallowAttrBox:
    """Lightweight holder for ``DataFrame.attrs`` entries that must not be deep-copied by Pandas."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> "_ShallowAttrBox":
        return self


def group_items_by_date(
    items: Sequence[pystac.Item],
    sensor: str,
    limit: int | None = None,
    bbox_4326: Sequence[float] | None = None,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_coverage_pct: float = 0.0,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Group STAC items by acquisition date (``YYYY-MM-DD``) for seamless tile mosaicking.

    Computes geometric AOI footprint coverage ``coverage_pct`` for every date using the
    union of all same-day tiles and optionally filters out partial-swath dates with
    ``coverage_pct < min_coverage_pct``.

    Returns ``(grouped_items, summary_df)`` ordered newest-first, where each entry
    in ``grouped_items`` is the list of all STAC items from that acquisition date
    covering the AOI (1-to-1 aligned with ``summary_df`` rows). Also attaches
    ``summary_df.attrs["items_by_date"] = {date_str: [item_1, item_2, ...]}``.
    """
    if not items:
        empty_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        empty_df.attrs["items_by_date"] = _NoDeepCopyDict()
        empty_df.attrs["total_dates_before_coverage_filter"] = 0
        empty_df.attrs["filtered_partial_dates"] = 0
        empty_df.attrs["total_dates_found"] = 0
        empty_df.attrs["selected_dates_count"] = 0
        empty_df.attrs["interval_mode"] = INTERVAL_ALL
        return [], empty_df

    buckets: dict[str, list[pystac.Item]] = {}
    for it in items:
        dt_utc = _extract_item_datetime_utc(it)
        date_str = dt_utc.strftime("%Y-%m-%d")
        buckets.setdefault(date_str, []).append(it)

    aoi_geom = _build_aoi_geom_4326(bbox_4326, aoi_polygon_4326)
    sorted_dates = sorted(buckets.keys(), reverse=True)
    total_before_cov = len(sorted_dates)
    filtered_partial = 0

    grouped_items: list[list[pystac.Item]] = []
    items_by_date: _NoDeepCopyDict = _NoDeepCopyDict()
    rows: list[dict[str, object]] = []

    for date_str in sorted_dates:
        day_items = buckets[date_str]
        cov_pct = _compute_group_aoi_coverage_pct(day_items, aoi_geom)
        if min_coverage_pct > 0.0 and cov_pct < float(min_coverage_pct):
            filtered_partial += 1
            continue

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
                "coverage_pct": cov_pct,
                "mgrs_tile": meta["mgrs_tile"],
                "platform": meta["platform"],
            }
        )

    if limit is not None and int(limit) > 0 and len(rows) > int(limit):
        keep_n = int(limit)
        rows = rows[:keep_n]
        grouped_items = grouped_items[:keep_n]
        keep_dates = {str(r["date_str"]) for r in rows}
        items_by_date = _NoDeepCopyDict({d: v for d, v in items_by_date.items() if d in keep_dates})

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_df.attrs["items_by_date"] = items_by_date
    summary_df.attrs["total_dates_before_coverage_filter"] = total_before_cov
    summary_df.attrs["filtered_partial_dates"] = filtered_partial
    summary_df.attrs["total_dates_found"] = len(summary_df)
    summary_df.attrs["selected_dates_count"] = len(summary_df)
    summary_df.attrs["interval_mode"] = INTERVAL_ALL
    return grouped_items, summary_df


def filter_by_temporal_interval(
    df_dates: pd.DataFrame,
    items_by_date: dict[str, list[pystac.Item]],
    interval_mode: str = INTERVAL_ALL,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Select the single best/clearest date per temporal bucket (14d / monthly / seasonal / yearly).

    For each bucket picks the date with the highest ``coverage_pct`` and lowest ``cloud_cover``.
    For ``yearly`` mode applies a subtle preference for growing-season/summer months (May–Sep)
    among full-coverage dates so multi-year vegetation/land-cover comparisons stay consistent.

    Returns ``(grouped_items, filtered_df)`` ordered newest-first.
    """
    mode = str(interval_mode or INTERVAL_ALL).strip().lower()
    if mode not in TEMPORAL_INTERVAL_KEYS:
        mode = INTERVAL_ALL

    base_total = int(len(df_dates))
    base_before_cov = int(df_dates.attrs.get("total_dates_before_coverage_filter", base_total))
    base_partial = int(df_dates.attrs.get("filtered_partial_dates", 0))

    if df_dates.empty:
        empty_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        empty_df.attrs["items_by_date"] = _NoDeepCopyDict()
        empty_df.attrs["total_dates_before_coverage_filter"] = base_before_cov
        empty_df.attrs["filtered_partial_dates"] = base_partial
        empty_df.attrs["total_dates_found"] = 0
        empty_df.attrs["selected_dates_count"] = 0
        empty_df.attrs["interval_mode"] = mode
        return [], empty_df

    # Extract lightweight plain records without triggering DataFrame.attrs deep copies
    records = df_dates.to_dict(orient="records")

    if mode == INTERVAL_ALL:
        valid_recs = [r for r in records if str(r["date_str"]) in items_by_date]
        valid_recs.sort(key=lambda r: str(r["date_str"]), reverse=True)
        ordered_df = pd.DataFrame(valid_recs, columns=SUMMARY_COLUMNS)
        grouped = [items_by_date[str(r["date_str"])] for r in valid_recs]
        ordered_df.attrs["items_by_date"] = _NoDeepCopyDict(
            {str(r["date_str"]): items_by_date[str(r["date_str"])] for r in valid_recs}
        )
        ordered_df.attrs["candidates_by_bucket"] = _NoDeepCopyDict(
            {str(r["date_str"]): [items_by_date[str(r["date_str"])]] for r in valid_recs}
        )
        ordered_df.attrs["total_dates_before_coverage_filter"] = base_before_cov
        ordered_df.attrs["filtered_partial_dates"] = base_partial
        ordered_df.attrs["total_dates_found"] = base_total
        ordered_df.attrs["selected_dates_count"] = len(ordered_df)
        ordered_df.attrs["interval_mode"] = mode
        return grouped, ordered_df

    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for rec in records:
        d_str = str(rec["date_str"])
        if d_str not in items_by_date:
            continue
        dt_val = date.fromisoformat(d_str[:10])
        if mode == INTERVAL_14D:
            doy = dt_val.timetuple().tm_yday
            bucket_key = (dt_val.year, (doy - 1) // 14)
        elif mode == INTERVAL_MONTHLY:
            bucket_key = (dt_val.year, dt_val.month)
        elif mode == INTERVAL_SEASONAL:
            bucket_key = (dt_val.year, (dt_val.month - 1) // 3)
        else:  # INTERVAL_YEARLY
            bucket_key = (dt_val.year, 0)

        rec_copy = dict(rec)
        rec_copy["_month"] = dt_val.month
        buckets.setdefault(bucket_key, []).append(rec_copy)

    selected_rows: list[dict[str, Any]] = []
    candidates_by_bucket: dict[str, list[list[pystac.Item]]] = {}
    for bucket_key in sorted(buckets.keys(), reverse=True):
        cands = buckets[bucket_key]

        def _cand_rank(r: dict[str, Any]) -> tuple[float, int, float, int, str]:
            cov_raw = r.get("coverage_pct", 100.0)
            cov = float(cov_raw) if pd.notna(cov_raw) else 100.0
            cld_raw = r.get("cloud_cover", np.nan)
            cld = float(cld_raw) if pd.notna(cld_raw) else 50.0
            month = int(r.get("_month", 7))
            if mode == INTERVAL_YEARLY:
                # Highest rounded coverage first, then subtle growing-season preference (May-Sep),
                # then lowest cloud cover, then closest to mid-July.
                summer_pen = 0 if (5 <= month <= 9) else 1
                return (-round(cov, 0), summer_pen, cld, abs(month - 7), str(r["date_str"]))
            return (-round(cov, 1), 0, cld, 0, str(r["date_str"]))

        ranked_cands = sorted(cands, key=_cand_rank)
        best = dict(ranked_cands[0])
        best.pop("_month", None)
        best_date = str(best["date_str"])
        selected_rows.append(best)
        # Keep top 4 clearest candidate dates per time bucket for AOI-level cloud fallback
        candidates_by_bucket[best_date] = [
            items_by_date[str(c["date_str"])]
            for c in ranked_cands[:4]
            if str(c["date_str"]) in items_by_date
        ]

    selected_rows.sort(key=lambda r: str(r["date_str"]), reverse=True)
    filtered_df = pd.DataFrame(selected_rows, columns=SUMMARY_COLUMNS)
    grouped_items = [items_by_date[str(r["date_str"])] for r in selected_rows]
    filtered_items_by_date = _NoDeepCopyDict(
        {str(r["date_str"]): items_by_date[str(r["date_str"])] for r in selected_rows}
    )

    filtered_df.attrs["items_by_date"] = filtered_items_by_date
    filtered_df.attrs["candidates_by_bucket"] = _NoDeepCopyDict(candidates_by_bucket)
    filtered_df.attrs["total_dates_before_coverage_filter"] = base_before_cov
    filtered_df.attrs["filtered_partial_dates"] = base_partial
    filtered_df.attrs["total_dates_found"] = base_total
    filtered_df.attrs["selected_dates_count"] = len(filtered_df)
    filtered_df.attrs["interval_mode"] = mode
    return grouped_items, filtered_df


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


def _finalize_search_results(
    raw_items: Sequence[pystac.Item],
    *,
    sensor: str,
    bbox_4326: Sequence[float],
    aoi_polygon_4326: dict[str, Any] | None,
    min_coverage_pct: float,
    interval_mode: str,
    limit: int | None,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Group by date, filter partial swaths, apply temporal interval, and attach raw archive metadata."""
    # Group all full-coverage dates first (without premature limit) so interval bucketing spans all years
    _, full_df = group_items_by_date(
        raw_items,
        sensor=sensor,
        limit=None,
        bbox_4326=bbox_4326,
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=float(min_coverage_pct),
    )
    raw_items_by_date = _NoDeepCopyDict(full_df.attrs.get("items_by_date", {}))

    grouped_items, final_df = filter_by_temporal_interval(
        full_df,
        raw_items_by_date,
        interval_mode=interval_mode,
    )

    if limit is not None and int(limit) > 0 and len(final_df) > int(limit):
        keep_n = int(limit)
        final_df = final_df.iloc[:keep_n].reset_index(drop=True)
        grouped_items = grouped_items[:keep_n]
        keep_dates = set(final_df["date_str"].astype(str))
        final_df.attrs["items_by_date"] = _NoDeepCopyDict(
            {d: v for d, v in final_df.attrs.get("items_by_date", {}).items() if d in keep_dates}
        )
        final_df.attrs["selected_dates_count"] = len(final_df)

    final_df.attrs["raw_summary_df"] = full_df
    final_df.attrs["raw_items_by_date"] = raw_items_by_date
    return grouped_items, final_df


# ---------------------------------------------------------------------------
# 1. SENTINEL-2 L2A
# ---------------------------------------------------------------------------
def search_sentinel2(
    bbox_4326: Sequence[float],
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    max_cloud_cover: float = 20,
    limit: int | None = None,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_coverage_pct: float = DEFAULT_MIN_AOI_COVERAGE_PCT,
    interval_mode: str = INTERVAL_ALL,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Search Sentinel-2 L2A scenes on Planetary Computer, grouped by date with coverage & interval filters."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    raw_items = _fetch_stac_items_full(
        collection=SENTINEL2_COLLECTION,
        bbox_4326=bbox_4326,
        start_date=start_date,
        end_date=end_date,
        query={"eo:cloud_cover": {"lte": float(max_cloud_cover)}},
        max_items_per_window=1000,
        page_limit=250,
    )
    return _finalize_search_results(
        raw_items,
        sensor=SENSOR_S2,
        bbox_4326=bbox_4326,
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=min_coverage_pct,
        interval_mode=interval_mode,
        limit=limit,
    )


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
    limit: int | None = None,
    exclude_l7_slc_off: bool = True,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_coverage_pct: float = DEFAULT_MIN_AOI_COVERAGE_PCT,
    interval_mode: str = INTERVAL_ALL,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Search Landsat 4/5/7/8/9 Collection 2 Level-2 scenes across the full 1984–present archive."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    raw_items = _fetch_stac_items_full(
        collection=LANDSAT_COLLECTION,
        bbox_4326=bbox_4326,
        start_date=start_date,
        end_date=end_date,
        query={
            "eo:cloud_cover": {"lte": float(max_cloud_cover)},
            "platform": {"in": LANDSAT_OPERATIONAL_PLATFORMS},
        },
        max_items_per_window=1000,
        page_limit=250,
    )
    if exclude_l7_slc_off:
        raw_items = [it for it in raw_items if not _is_landsat7_slc_off(it)]

    return _finalize_search_results(
        raw_items,
        sensor=SENSOR_LANDSAT,
        bbox_4326=bbox_4326,
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=min_coverage_pct,
        interval_mode=interval_mode,
        limit=limit,
    )


def calibrate_landsat_optical_dn(
    raw_vals: np.ndarray,
    qa_vals: np.ndarray | None = None,
) -> np.ndarray:
    """Calibrate raw Landsat C2 L2 optical DN values to clean surface reflectance in ``[0.0, 1.0]``.

    Masks raw fill (``DN == 0``), radiometrically saturated/corrupt pixels (``DN >= 65455``),
    and ``qa_pixel`` bit 0 (Fill) to ``np.nan`` BEFORE applying ``DN * 0.0000275 - 0.2``,
    eliminating bright red/green/blue sensor noise speckles on 1980s Landsat 4/5 scenes.
    """
    arr = np.asarray(raw_vals)
    valid_mask = np.isfinite(arr) & (arr > 0) & (arr < 65455)
    if qa_vals is not None:
        qa_u16 = np.asarray(qa_vals, dtype=np.uint16)
        valid_mask &= (qa_u16 & np.uint16(1)) == 0

    refl = np.full(arr.shape, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        scaled = arr[valid_mask].astype(np.float32) * np.float32(LANDSAT_SR_SCALE) + np.float32(LANDSAT_SR_OFFSET)
        np.clip(scaled, np.float32(0.0), np.float32(1.0), out=scaled)
        refl[valid_mask] = scaled
    return refl


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
    masking fill (``DN == 0``), saturated (``DN >= 65455``), and ``qa_pixel`` Fill bit 0
    pixels to ``NaN`` before scaling.
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

    # Always stream qa_pixel internally when loading optical bands so bit-0 Fill speckles are masked
    user_requested_qa = "qa_pixel" in asset_list
    has_optical = any(a != "qa_pixel" for a in asset_list)
    load_assets = list(asset_list)
    if has_optical and "qa_pixel" not in load_assets:
        load_assets.append("qa_pixel")

    item_list = _normalize_item_group(item)
    crs_str = _normalize_crs_str(target_epsg)
    signed_items = [planetary_computer.sign(it) for it in item_list]

    if all(a == "qa_pixel" for a in load_assets):
        resampling_cfg: str | dict[str, str] = "nearest"
    elif "qa_pixel" in load_assets:
        resampling_cfg = {"qa_pixel": "nearest", "*": "bilinear"}
    else:
        resampling_cfg = "bilinear"

    _configure_odc_rio()

    raw_ds = odc.stac.load(
        signed_items,
        bands=load_assets,
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling=resampling_cfg,
        groupby="solar_day",
    )
    raw_ds = _collapse_time_dim(raw_ds, sensor=SENSOR_LANDSAT)

    # Prepare qa_pixel array first so optical bands can mask Fill bit 0 cleanly
    qa_vals_u16: np.ndarray | None = None
    if "qa_pixel" in raw_ds.data_vars:
        qa_raw = raw_ds["qa_pixel"].values
        qa_vals_u16 = np.where(np.isfinite(qa_raw), qa_raw, 1).astype(np.uint16)

    calibrated_vars: dict[str, xr.DataArray] = {}
    for asset_name in asset_list:
        if asset_name not in raw_ds.data_vars:
            raise KeyError(
                f"Asset {asset_name!r} was not loaded from Landsat item(s) {[it.id for it in signed_items]!r}."
            )

        logical_name = LANDSAT_ASSET_TO_LOGICAL.get(asset_name, asset_name)
        raw_da = raw_ds[asset_name]

        if asset_name == "qa_pixel":
            if not user_requested_qa:
                continue
            qa_vals = qa_vals_u16 if qa_vals_u16 is not None else np.ones(raw_da.shape, dtype=np.uint16)
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
            refl = calibrate_landsat_optical_dn(raw_vals, qa_vals=qa_vals_u16)

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
            del raw_vals, refl

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
    limit: int | None = None,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_coverage_pct: float = DEFAULT_MIN_AOI_COVERAGE_PCT,
    interval_mode: str = INTERVAL_ALL,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Search Sentinel-1 SAR RTC scenes on Planetary Computer, grouped by date with coverage & interval filters."""
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must contain 4 values [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    raw_items = _fetch_stac_items_full(
        collection=SENTINEL1_COLLECTION,
        bbox_4326=bbox_4326,
        start_date=start_date,
        end_date=end_date,
        query=None,
        max_items_per_window=1000,
        page_limit=250,
    )
    return _finalize_search_results(
        raw_items,
        sensor=SENSOR_S1,
        bbox_4326=bbox_4326,
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=min_coverage_pct,
        interval_mode=interval_mode,
        limit=limit,
    )


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
    limit: int | None = None,
    exclude_l7_slc_off: bool = True,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_coverage_pct: float = DEFAULT_MIN_AOI_COVERAGE_PCT,
    interval_mode: str = INTERVAL_ALL,
) -> tuple[list[list[pystac.Item]], pd.DataFrame]:
    """Dispatch full-archive date-grouped STAC search with coverage & temporal interval filters."""
    if sensor == SENSOR_S2:
        return search_sentinel2(
            bbox_4326=bbox_4326,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
            aoi_polygon_4326=aoi_polygon_4326,
            min_coverage_pct=min_coverage_pct,
            interval_mode=interval_mode,
        )
    if sensor == SENSOR_LANDSAT:
        return search_landsat(
            bbox_4326=bbox_4326,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
            exclude_l7_slc_off=exclude_l7_slc_off,
            aoi_polygon_4326=aoi_polygon_4326,
            min_coverage_pct=min_coverage_pct,
            interval_mode=interval_mode,
        )
    if sensor == SENSOR_S1:
        return search_sentinel1(
            bbox_4326=bbox_4326,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            aoi_polygon_4326=aoi_polygon_4326,
            min_coverage_pct=min_coverage_pct,
            interval_mode=interval_mode,
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


def load_copernicus_dem(
    bbox_4326: Sequence[float],
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 30.0,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.DataArray | None:
    """Stream Copernicus DEM 30 m (``cop-dem-glo-30``) clipped to the AOI in ``target_epsg``.

    Returns a float32 elevation ``xr.DataArray`` [m a.s.l.] with invalid voids
    (``<= -500 m``) masked to ``np.nan``, or ``None`` if no tiles cover the AOI.
    """
    try:
        catalog = get_stac_catalog()
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=list(bbox_4326),
            max_items=8,
        )
        items = list(search.items())
        if not items:
            return None

        ds = odc.stac.load(
            items,
            bands=["data"],
            bbox=list(bbox_4326),
            crs=str(target_epsg),
            resolution=float(resolution),
            groupby="solar_day",
        )
        da = ds["data"].isel(time=0, drop=True).compute().astype(np.float32)
        da = da.where(da > -500.0)
        da = da.rio.write_crs(str(target_epsg), inplace=False)
        da = da.rio.write_nodata(np.nan, inplace=False)
        if aoi_polygon_4326 is not None:
            da = clip_to_aoi_polygon(da, aoi_polygon_4326, drop=False)
        da = da.assign_attrs(
            {
                "long_name": "Copernicus DEM GLO-30 Elevation",
                "units": "m",
                "source": "cop-dem-glo-30",
                "resolution_m": float(resolution),
            }
        )
        del ds
        gc.collect()
        return da
    except Exception as exc:
        logger.warning("Copernicus DEM 30 m streaming unavailable for bbox %s: %s", bbox_4326, exc)
        return None

