"""Geodetic coordinate transformations, AOI builders, and GeoTIFF export utilities.

Focused on Slovenia's national coordinate reference system D96/TM (EPSG:3794),
with full support for WGS84 (EPSG:4326) and WGS 84 / UTM zone 33N (EPSG:32633).
"""

from __future__ import annotations

import gc
import os
from functools import lru_cache
from typing import Sequence

# Ensure GDAL/rasterio honors custom system/proxy CA bundles before library init
_ca_bundle = os.environ.get("CURL_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
if _ca_bundle and os.path.exists(_ca_bundle):
    os.environ.setdefault("GDAL_CURL_CA_BUNDLE", _ca_bundle)

import numpy as np
import rasterio
import rioxarray  # noqa: F401 - registers the .rio accessor on xarray objects
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

# ---------------------------------------------------------------------------
# Supported CRS definitions
# ---------------------------------------------------------------------------
EPSG_D96TM: int = 3794
EPSG_WGS84: int = 4326
EPSG_UTM33N: int = 32633
EPSG_POLAR_NORTH: int = 3413
EPSG_POLAR_SOUTH: int = 3031

CRS_D96TM: str = "EPSG:3794"
CRS_WGS84: str = "EPSG:4326"
CRS_UTM33N: str = "EPSG:32633"
CRS_POLAR_NORTH: str = "EPSG:3413"
CRS_POLAR_SOUTH: str = "EPSG:3031"

SUPPORTED_CRS: dict[str, dict[str, object]] = {
    CRS_D96TM: {
        "epsg": EPSG_D96TM,
        "name": "Slovenia 1996 / Slovene National Grid (D96/TM)",
        "short_name": "D96/TM",
        "units": "m",
        "axes": ("Easting (e)", "Northing (n)"),
    },
    CRS_WGS84: {
        "epsg": EPSG_WGS84,
        "name": "WGS 84 (Geographic)",
        "short_name": "WGS84",
        "units": "deg",
        "axes": ("Longitude (lon)", "Latitude (lat)"),
    },
    CRS_UTM33N: {
        "epsg": EPSG_UTM33N,
        "name": "WGS 84 / UTM zone 33N",
        "short_name": "UTM 33N",
        "units": "m",
        "axes": ("Easting (E)", "Northing (N)"),
    },
    CRS_POLAR_NORTH: {
        "epsg": EPSG_POLAR_NORTH,
        "name": "WGS 84 / NSIDC Sea Ice Polar Stereographic North",
        "short_name": "Polar North",
        "units": "m",
        "axes": ("X [m]", "Y [m]"),
    },
    CRS_POLAR_SOUTH: {
        "epsg": EPSG_POLAR_SOUTH,
        "name": "WGS 84 / Antarctic Polar Stereographic",
        "short_name": "Polar South",
        "units": "m",
        "axes": ("X [m]", "Y [m]"),
    },
}

# RAM-safe resolution step-down thresholds (km^2)
# High-RAM profile (GitHub Codespaces 8-16 GB, Hugging Face Spaces 16 GB, Local PC):
_RES_10M_MAX_KM2: float = 1200.0
_RES_20M_MAX_KM2: float = 2500.0
_RES_30M_MAX_KM2: float = 4000.0
_HIGH_RAM_MAX_RADIUS_M: int = 31500

# Low-RAM safety profile (Streamlit Community Cloud 1 GB container):
_CLOUD_1GB_RES_10M_MAX_KM2: float = 150.0
_CLOUD_1GB_RES_20M_MAX_KM2: float = 400.0
_CLOUD_1GB_RES_30M_MAX_KM2: float = 900.0
_CLOUD_1GB_MAX_RADIUS_M: int = 15000


def is_codespaces() -> bool:
    """Return True when running inside a GitHub Codespaces container (8–16 GB RAM)."""
    return str(os.environ.get("CODESPACES", "")).strip().lower() == "true"


def _detect_cgroup_ram_gb() -> float | None:
    """Read Linux cgroup v2/v1 container memory limit in GiB when available."""
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            if not os.path.exists(path):
                continue
            raw = open(path, "r", encoding="utf-8").read().strip()
            if not raw or raw == "max":
                continue
            val = int(raw)
            # Ignore huge/unbounded sentinel values (> 1 TiB)
            if 128 * 1024 * 1024 <= val <= 1024 * 1024 * 1024 * 1024:
                return val / (1024.0 ** 3)
        except Exception:
            continue
    return None


def is_low_ram_cloud() -> bool:
    """Return True when running inside a constrained (~1 GB RAM) Streamlit Community Cloud container."""
    if is_codespaces():
        return False
    if str(os.environ.get("SPACE_ID", "")).strip():
        return False
    if str(os.environ.get("STREAMLIT_SHARING_MODE", "")).strip():
        return True
    try:
        if "/mount/src/" in os.getcwd().replace("\\", "/"):
            return True
    except Exception:
        pass
    cgroup_gb = _detect_cgroup_ram_gb()
    if cgroup_gb is not None and cgroup_gb <= 2.5:
        return True
    return False


def get_runtime_environment() -> dict[str, object]:
    """Return active runtime profile (Codespaces / HF Spaces / Streamlit Cloud 1 GB / Local)."""
    if is_codespaces():
        return {
            "mode": "codespaces",
            "high_ram": True,
            "max_workers": 4,
            "max_area_km2": _RES_30M_MAX_KM2,
            "max_radius_m": _HIGH_RAM_MAX_RADIUS_M,
            "res_10m_max_km2": _RES_10M_MAX_KM2,
            "res_20m_max_km2": _RES_20M_MAX_KM2,
        }
    if str(os.environ.get("SPACE_ID", "")).strip():
        return {
            "mode": "hf_spaces",
            "high_ram": True,
            "max_workers": 4,
            "max_area_km2": _RES_30M_MAX_KM2,
            "max_radius_m": _HIGH_RAM_MAX_RADIUS_M,
            "res_10m_max_km2": _RES_10M_MAX_KM2,
            "res_20m_max_km2": _RES_20M_MAX_KM2,
        }
    if is_low_ram_cloud():
        return {
            "mode": "streamlit_cloud",
            "high_ram": False,
            "max_workers": 2,
            "max_area_km2": _CLOUD_1GB_RES_30M_MAX_KM2,
            "max_radius_m": _CLOUD_1GB_MAX_RADIUS_M,
            "res_10m_max_km2": _CLOUD_1GB_RES_10M_MAX_KM2,
            "res_20m_max_km2": _CLOUD_1GB_RES_20M_MAX_KM2,
        }
    return {
        "mode": "local",
        "high_ram": True,
        "max_workers": 4,
        "max_area_km2": _RES_30M_MAX_KM2,
        "max_radius_m": _HIGH_RAM_MAX_RADIUS_M,
        "res_10m_max_km2": _RES_10M_MAX_KM2,
        "res_20m_max_km2": _RES_20M_MAX_KM2,
    }


def get_ram_limits(low_ram_override: bool | None = None) -> dict[str, object]:
    """Return hard RAM-based AOI limits (``max_area_km2``, ``max_radius_m``, resolution thresholds).

    - Streamlit Community Cloud (1 GB RAM): ``max_area_km2 = 900.0``, ``max_radius_m = 15000``
      (30 km × 30 km = 900 km² exact square cap).
    - GitHub Codespaces / HF Spaces / Local (>= 8 GB RAM): ``max_area_km2 = 4000.0``,
      ``max_radius_m = 31500`` (~3969 km²).
    """
    if low_ram_override is True:
        return {
            "mode": "streamlit_cloud",
            "high_ram": False,
            "max_area_km2": _CLOUD_1GB_RES_30M_MAX_KM2,
            "max_radius_m": _CLOUD_1GB_MAX_RADIUS_M,
            "res_10m_max_km2": _CLOUD_1GB_RES_10M_MAX_KM2,
            "res_20m_max_km2": _CLOUD_1GB_RES_20M_MAX_KM2,
            "max_workers": 2,
        }
    if low_ram_override is False:
        return {
            "mode": "high_ram",
            "high_ram": True,
            "max_area_km2": _RES_30M_MAX_KM2,
            "max_radius_m": _HIGH_RAM_MAX_RADIUS_M,
            "res_10m_max_km2": _RES_10M_MAX_KM2,
            "res_20m_max_km2": _RES_20M_MAX_KM2,
            "max_workers": 4,
        }
    return dict(get_runtime_environment())


def get_dynamic_aoi_limits(
    sensor: str | None = None,
    resolution_m: float | int = 10,
    low_ram_override: bool | None = None,
) -> dict[str, object]:
    """Compute resolution-aware pixel-budget AOI limits (``max_radius_m``, ``max_area_km2``, ``max_pixels``).

    On 1 GB Streamlit Cloud, caps per-band pixels at ~1,000,000 px (4,000,000 px on High-RAM):
    - Sentinel-2 / Landsat / S1 at 10–30 m: up to 15 km radius (~900 km²) on 1 GB Cloud
    - Sentinel-2 / Landsat at 100 m / 250 m overviews: up to 50–80 km radius (~10,000–25,600 km²)
    - MODIS at 500 m: up to 250 km radius (~250,000 km²)
    - MODIS at 1000 m / 2000 m: up to 600–1200 km radius (1.44–5.76M km², entire Greenland / Amazon / West Antarctica)
    """
    base = get_ram_limits(low_ram_override=low_ram_override)
    high_ram = bool(base["high_ram"])
    s_norm = str(sensor or "sentinel-2-l2a").strip().lower()
    is_modis = s_norm.startswith("modis")
    res_f = max(10.0, float(resolution_m or (500.0 if is_modis else 10.0)))
    max_pixels = 4_000_000 if high_ram else 1_000_000

    if is_modis:
        if res_f >= 2000.0:
            max_rad = 1_600_000 if high_ram else 1_200_000
        elif res_f >= 1000.0:
            max_rad = 1_200_000 if high_ram else 600_000
        else:
            max_rad = 500_000 if high_ram else 250_000
        max_area = float((2.0 * max_rad) ** 2) / 1e6
        return {
            "sensor": s_norm,
            "resolution_m": int(round(res_f)),
            "high_ram": high_ram,
            "max_pixels": max_pixels,
            "max_radius_m": int(max_rad),
            "max_area_km2": round(max_area, 1),
            "is_continental": True,
        }

    # Optical / SAR regional & local sensors
    if res_f >= 250.0:
        max_rad = 120_000 if high_ram else 80_000
    elif res_f >= 100.0:
        max_rad = 80_000 if high_ram else 50_000
    elif res_f >= 60.0:
        max_rad = 60_000 if high_ram else 30_000
    else:
        max_rad = int(base["max_radius_m"])
    max_area = float((2.0 * max_rad) ** 2) / 1e6
    return {
        "sensor": s_norm,
        "resolution_m": int(round(res_f)),
        "high_ram": high_ram,
        "max_pixels": max_pixels,
        "max_radius_m": int(max_rad),
        "max_area_km2": round(max_area, 1),
        "is_continental": False,
    }


def get_safe_processing_resolution(
    requested_res_m: int | float,
    area_km2: float,
    sensor: str | None = None,
    low_ram_override: bool | None = None,
) -> tuple[int, bool]:
    """Return ``(effective_resolution_m, was_adjusted)`` so large valid AOIs never crash 1 GB RAM.

    Respects MODIS continental resolutions (500 m / 1000 m / 2000 m), Landsat >= 30 m,
    and automatically bumps resolution upward when pixel count exceeds the RAM budget.
    """
    s_norm = str(sensor or "").strip().lower()
    is_modis = s_norm.startswith("modis")
    is_modis_lst = s_norm == "modis-11a2-061"
    is_landsat = s_norm == "landsat-c2-l2"

    req = max(10, int(round(float(requested_res_m))))
    if is_modis_lst and req < 1000:
        req = 1000
    elif is_modis and req < 500:
        req = 500
    elif is_landsat and req < 30:
        req = 30

    limits = get_ram_limits(low_ram_override=low_ram_override)
    high_ram = bool(limits["high_ram"])

    if is_modis:
        max_px = 4_000_000 if high_ram else 1_450_000
        est_px = (float(area_km2) * 1e6) / float(req * req)
        if est_px <= max_px:
            return int(req), False
        for cand in (500, 1000, 2000, 4000):
            if cand >= req and (float(area_km2) * 1e6) / float(cand * cand) <= max_px:
                return int(cand), (int(cand) > int(req))
        return 2000, (2000 > int(req))

    if high_ram:
        return req, False

    if area_km2 > 10_000.0:
        safe = max(req, 250)
    elif area_km2 > 3_600.0:
        safe = max(req, 100)
    elif area_km2 > float(limits["max_area_km2"]) * 1.02:
        safe = max(req, 60)
    elif area_km2 > float(limits["res_20m_max_km2"]):
        safe = max(req, 30)
    elif area_km2 > float(limits["res_10m_max_km2"]):
        safe = max(req, 20)
    else:
        safe = req
    return int(safe), (int(safe) > int(req))


def get_parallel_workers() -> int:
    """Return optimal worker thread count (4 on Codespaces/HF/local, 2 on 1 GB Streamlit Cloud)."""
    return int(get_runtime_environment()["max_workers"])


def normalize_crs(epsg_or_crs: int | str | CRS) -> CRS:
    """Normalize an EPSG integer, CRS string, or pyproj.CRS into a pyproj.CRS."""
    if isinstance(epsg_or_crs, CRS):
        return epsg_or_crs
    if isinstance(epsg_or_crs, int):
        return CRS.from_epsg(epsg_or_crs)
    return CRS.from_user_input(epsg_or_crs)


@lru_cache(maxsize=32)
def _get_transformer(src_key: str, dst_key: str) -> Transformer:
    """Return a cached pyproj Transformer using strict (x, y) / (lon, lat) axis order."""
    return Transformer.from_crs(
        CRS.from_user_input(src_key),
        CRS.from_user_input(dst_key),
        always_xy=True,
    )


def _crs_cache_key(crs: CRS) -> str:
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs.to_wkt()


def transform_coords(
    x: float,
    y: float,
    src_epsg: int | str | CRS,
    dst_epsg: int | str | CRS,
) -> tuple[float, float]:
    """Transform a single 2D coordinate pair exactly between two CRS definitions.

    Uses ``always_xy=True`` so coordinate order is always ``(lon, lat)`` for
    geographic CRS (e.g., EPSG:4326) and ``(easting, northing)`` for projected
    CRS (e.g., EPSG:3794, EPSG:32633).

    Parameters
    ----------
    x, y : float
        Input coordinates in ``src_epsg`` (lon/lat or easting/northing).
    src_epsg : int | str | pyproj.CRS
        Source CRS (e.g., ``4326`` or ``"EPSG:4326"``).
    dst_epsg : int | str | pyproj.CRS
        Destination CRS (e.g., ``3794`` or ``"EPSG:3794"``).

    Returns
    -------
    tuple[float, float]
        Transformed ``(dst_x, dst_y)`` coordinates.
    """
    src_crs = normalize_crs(src_epsg)
    dst_crs = normalize_crs(dst_epsg)
    if src_crs == dst_crs:
        return (float(x), float(y))

    transformer = _get_transformer(_crs_cache_key(src_crs), _crs_cache_key(dst_crs))
    dst_x, dst_y = transformer.transform(float(x), float(y))
    return (float(dst_x), float(dst_y))


def is_in_slovenia(lat: float, lon: float) -> bool:
    """Return ``True`` if ``(lat, lon)`` lies within Slovenia's D96/TM validity bounds."""
    lat_f = float(lat)
    lon_f = float(lon)
    return (45.35 <= lat_f <= 46.95) and (13.30 <= lon_f <= 16.65)


def get_auto_utm_epsg(lat: float, lon: float) -> tuple[int, str]:
    """Compute the exact local WGS 84 / UTM EPSG code and human-readable label anywhere on Earth.

    Uses ``zone = int((lon + 180) // 6) + 1`` clamped to ``[1, 60]``, returning
    ``(32600 + zone, "WGS 84 / UTM {zone}N (EPSG:326xx)")`` in the Northern Hemisphere
    (``lat >= 0``) or ``(32700 + zone, "WGS 84 / UTM {zone}S (EPSG:327xx)")`` in the
    Southern Hemisphere (``lat < 0``).
    """
    lat_f = float(lat)
    lon_f = float(lon)
    if not (-90.0 <= lat_f <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}.")
    if not (-180.0 <= lon_f <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}.")

    zone = max(1, min(60, int((lon_f + 180.0) // 6) + 1))
    if lat_f >= 0.0:
        epsg = 32600 + zone
        hemi = "N"
    else:
        epsg = 32700 + zone
        hemi = "S"
    return epsg, f"WGS 84 / UTM {zone}{hemi} (EPSG:{epsg})"


def get_auto_metric_epsg(lat: float, lon: float) -> int:
    """Return the optimal projected metric EPSG integer anywhere on Earth, including both poles.

    - Slovenia (``is_in_slovenia``): ``3794`` (D96/TM)
    - Arctic / Greenland (``lat >= 70.0``): ``3413`` (WGS 84 / NSIDC Sea Ice Polar Stereographic North)
    - Antarctica (``lat <= -65.0``): ``3031`` (WGS 84 / Antarctic Polar Stereographic)
    - Everywhere else: exact local Auto-UTM (``326xx`` / ``327xx``).
    """
    lat_f = float(lat)
    lon_f = float(lon)
    if is_in_slovenia(lat_f, lon_f):
        return EPSG_D96TM
    if lat_f >= 70.0:
        return EPSG_POLAR_NORTH
    if lat_f <= -65.0:
        return EPSG_POLAR_SOUTH
    epsg, _ = get_auto_utm_epsg(lat_f, lon_f)
    return int(epsg)


def get_best_metric_crs(
    lat: float,
    lon: float,
    mode: str = "auto",
) -> tuple[str, str]:
    """Resolve the active local metric CRS string and display name for ``(lat, lon)``.

    Parameters
    ----------
    lat, lon : float
        AOI center coordinates in WGS84 degrees.
    mode : str, default ``"auto"``
        - ``"auto"``: ``EPSG:3794`` (D96/TM) inside Slovenia, ``EPSG:3413`` in Arctic/Greenland
          (``lat >= 70°``), ``EPSG:3031`` in Antarctica (``lat <= -65°``), and local Auto-UTM elsewhere.
        - ``"d96"``: Force ``EPSG:3794`` (D96/TM).
        - ``"utm"``: Force exact local UTM (``EPSG:326xx`` / ``EPSG:327xx``).

    Returns
    -------
    tuple[str, str]
        ``(crs_str, display_name)``, e.g. ``("EPSG:3413", "Arctic Polar Stereographic (EPSG:3413)")``.
    """
    m = str(mode or "auto").strip().lower()
    if m == "d96":
        return CRS_D96TM, "D96/TM (EPSG:3794)"
    if m == "utm":
        epsg, name = get_auto_utm_epsg(lat, lon)
        return f"EPSG:{epsg}", name
    epsg_auto = get_auto_metric_epsg(lat, lon)
    crs_s = f"EPSG:{epsg_auto}"
    return crs_s, format_crs_label(crs_s, short=False)


def format_crs_label(
    epsg_or_crs: int | str | CRS,
    *,
    short: bool = False,
) -> str:
    """Return a clean human-readable label for D96/TM, Polar Stereographic (3413/3031), or any global UTM EPSG."""
    crs_obj = normalize_crs(epsg_or_crs)
    epsg = crs_obj.to_epsg()
    if epsg == EPSG_D96TM:
        return "D96/TM (3794)" if short else "D96/TM (EPSG:3794)"
    if epsg == EPSG_POLAR_NORTH:
        return "Polar North (3413)" if short else "Arctic Polar Stereographic (EPSG:3413)"
    if epsg == EPSG_POLAR_SOUTH:
        return "Polar South (3031)" if short else "Antarctic Polar Stereographic (EPSG:3031)"
    if epsg is not None and 32601 <= epsg <= 32660:
        zone = epsg - 32600
        return f"UTM {zone}N ({epsg})" if short else f"WGS 84 / UTM {zone}N (EPSG:{epsg})"
    if epsg is not None and 32701 <= epsg <= 32760:
        zone = epsg - 32700
        return f"UTM {zone}S ({epsg})" if short else f"WGS 84 / UTM {zone}S (EPSG:{epsg})"
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs_obj.name


def _metric_box_to_wgs84_bbox(
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    metric_crs: int | str | CRS = CRS_D96TM,
) -> list[float]:
    """Convert a projected metric bounding box to its enclosing EPSG:4326 bounding box.

    Edges are densified before reprojection so meridian convergence and
    projection curvature are accurately captured.
    """
    geom_metric = box(min_e, min_n, max_e, max_n)
    max_seg = max((max_e - min_e) / 16.0, (max_n - min_n) / 16.0, 50.0)
    geom_metric = geom_metric.segmentize(max_segment_length=max_seg)

    src_key = _crs_cache_key(normalize_crs(metric_crs))
    transformer = _get_transformer(src_key, CRS_WGS84)
    geom_4326 = shapely_transform(transformer.transform, geom_metric)
    min_lon, min_lat, max_lon, max_lat = geom_4326.bounds
    return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


def _d96tm_box_to_wgs84_bbox(
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
) -> list[float]:
    """Convert an EPSG:3794 bounding box to its enclosing EPSG:4326 bounding box."""
    return _metric_box_to_wgs84_bbox(min_e, min_n, max_e, max_n, metric_crs=CRS_D96TM)


def bbox_from_point_d96tm(
    e: float,
    n: float,
    radius_m: float,
) -> tuple[list[float], list[float]]:
    """Build an exact square AOI around Slovenian D96/TM coordinates ``(e, n)``.

    Parameters
    ----------
    e : float
        Easting in EPSG:3794 (meters).
    n : float
        Northing in EPSG:3794 (meters).
    radius_m : float
        Half-width of the square AOI in meters (total side length = ``2 * radius_m``).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(bbox_4326, bbox_3794)`` where:
        - ``bbox_4326 = [min_lon, min_lat, max_lon, max_lat]`` (for STAC search)
        - ``bbox_3794 = [min_e, min_n, max_e, max_n]`` (exact metric bounds)
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be strictly positive, got {radius_m}.")

    e_val = float(e)
    n_val = float(n)
    r_val = float(radius_m)

    bbox_3794 = [e_val - r_val, n_val - r_val, e_val + r_val, n_val + r_val]
    bbox_4326 = _d96tm_box_to_wgs84_bbox(*bbox_3794)
    return bbox_4326, bbox_3794


def bbox_from_point_wgs84(
    lat: float,
    lon: float,
    radius_m: float,
) -> tuple[list[float], list[float]]:
    """Build an exact distortion-free square AOI around a WGS84 point ``(lat, lon)``.

    Uses ``EPSG:3794`` (D96/TM) when inside Slovenia and the exact local WGS 84 / UTM
    zone (``EPSG:326xx`` / ``EPSG:327xx``) anywhere else on Earth, buffers by exact
    metric ``radius_m``, and returns ``(bbox_4326, bbox_metric)``.
    """
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}.")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}.")
    if radius_m <= 0:
        raise ValueError(f"radius_m must be strictly positive, got {radius_m}.")

    metric_crs, _ = get_best_metric_crs(float(lat), float(lon), mode="auto")
    e, n = transform_coords(float(lon), float(lat), EPSG_WGS84, metric_crs)
    r_val = float(radius_m)
    bbox_metric = [e - r_val, n - r_val, e + r_val, n + r_val]
    bbox_4326 = _metric_box_to_wgs84_bbox(*bbox_metric, metric_crs=metric_crs)
    return bbox_4326, bbox_metric


def metric_bbox_from_point(
    lat: float,
    lon: float,
    radius_m: float,
    target_epsg: int | str | None = None,
) -> tuple[list[float], str]:
    """Return an exact axis-aligned upright square ``[min_e, min_n, max_e, max_n]`` in ``target_epsg``."""
    if target_epsg is None:
        metric_crs, _ = get_best_metric_crs(float(lat), float(lon), mode="auto")
    else:
        metric_crs = f"EPSG:{target_epsg}" if isinstance(target_epsg, int) else str(target_epsg)
    e, n = transform_coords(float(lon), float(lat), EPSG_WGS84, metric_crs)
    r_val = float(radius_m)
    return [e - r_val, n - r_val, e + r_val, n + r_val], metric_crs


def inscribed_metric_bbox_from_wgs84(
    bbox_4326: Sequence[float],
    target_epsg: int | str,
) -> list[float]:
    """Compute the largest upright axis-aligned rectangle in ``target_epsg`` inscribed inside ``bbox_4326``.

    Eliminates ~1–3° meridian-convergence tilt wedges at raster corners when loading
    a WGS84 bounding box into a metric CRS (``EPSG:3794`` or local UTM).
    """
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
    crs_str = f"EPSG:{target_epsg}" if isinstance(target_epsg, int) else str(target_epsg)
    e_sw, n_sw = transform_coords(min_lon, min_lat, EPSG_WGS84, crs_str)
    e_se, n_se = transform_coords(max_lon, min_lat, EPSG_WGS84, crs_str)
    e_nw, n_nw = transform_coords(min_lon, max_lat, EPSG_WGS84, crs_str)
    e_ne, n_ne = transform_coords(max_lon, max_lat, EPSG_WGS84, crs_str)

    min_e = max(e_sw, e_nw)
    max_e = min(e_se, e_ne)
    min_n = max(n_sw, n_se)
    max_n = min(n_nw, n_ne)
    if max_e <= min_e or max_n <= min_n:
        min_e = min(e_sw, e_nw, e_se, e_ne)
        max_e = max(e_sw, e_nw, e_se, e_ne)
        min_n = min(n_sw, n_se, n_nw, n_ne)
        max_n = max(n_sw, n_se, n_nw, n_ne)
    return [float(min_e), float(min_n), float(max_e), float(max_n)]


def _recommended_resolution_for_area(
    area_km2: float,
    sensor: str | None = None,
) -> int:
    """Return a RAM-safe target spatial resolution (10–2000 m) for the active sensor and runtime."""
    s_norm = str(sensor or "").strip().lower()
    if s_norm.startswith("modis"):
        if s_norm == "modis-11a2-061":
            return 1000 if area_km2 <= 2_500_000.0 else 2000
        if area_km2 <= 260_000.0:
            return 500
        if area_km2 <= 2_500_000.0:
            return 1000
        return 2000
    env = get_runtime_environment()
    max_10m = float(env["res_10m_max_km2"])
    max_20m = float(env["res_20m_max_km2"])
    max_30m = float(env["max_area_km2"])
    if s_norm == "landsat-c2-l2":
        if area_km2 <= max_30m * 1.02:
            return 30
        if area_km2 <= 3_600.0:
            return 60
        return 100 if area_km2 <= 10_000.0 else 250
    if area_km2 <= max_10m:
        return 10
    if area_km2 <= max_20m:
        return 20
    if area_km2 <= max_30m * 1.02:
        return 30
    if area_km2 <= 3_600.0:
        return 60
    return 100 if area_km2 <= 10_000.0 else 250


def validate_aoi_size(
    bbox_4326: Sequence[float],
    max_area_km2: float | None = None,
    sensor: str | None = None,
    resolution_m: float | int | None = None,
) -> tuple[float, bool, int]:
    """Validate an EPSG:4326 AOI bounding box by computing exact metric area worldwide.

    Computes exact area in ``EPSG:3794`` (Slovenia), Polar Stereographic ``EPSG:3413`` / ``EPSG:3031``
    (Arctic / Antarctica), or local Auto-UTM elsewhere, using pixel-budget limits for MODIS and high-res sensors.
    """
    if len(bbox_4326) != 4:
        raise ValueError(
            f"bbox_4326 must have 4 elements [min_lon, min_lat, max_lon, max_lat], got {len(bbox_4326)}."
        )

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox_4326)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(
            f"Invalid bbox_4326 ordering: min must be strictly less than max, got {bbox_4326}."
        )

    if max_area_km2 is not None:
        eff_max_area = float(max_area_km2)
    elif sensor and str(sensor).strip().lower().startswith("modis"):
        eff_res = float(resolution_m) if resolution_m is not None else 2000.0
        eff_max_area = float(get_dynamic_aoi_limits(sensor, eff_res)["max_area_km2"])
    elif resolution_m is not None and float(resolution_m) >= 60.0:
        eff_max_area = float(get_dynamic_aoi_limits(sensor, float(resolution_m))["max_area_km2"])
    else:
        eff_max_area = float(get_runtime_environment()["max_area_km2"])

    geom_4326 = box(min_lon, min_lat, max_lon, max_lat)
    max_seg = max((max_lon - min_lon) / 16.0, (max_lat - min_lat) / 16.0, 1e-4)
    geom_4326 = geom_4326.segmentize(max_segment_length=max_seg)

    c_lat = 0.5 * (min_lat + max_lat)
    c_lon = 0.5 * (min_lon + max_lon)
    metric_crs, _ = get_best_metric_crs(c_lat, c_lon, mode="auto")
    transformer = _get_transformer(CRS_WGS84, metric_crs)
    geom_metric = shapely_transform(transformer.transform, geom_4326)
    area_km2 = float(geom_metric.area / 1_000_000.0)

    is_valid = (area_km2 > 0.0) and (area_km2 <= eff_max_area * 1.05)
    recommended_resolution_m = _recommended_resolution_for_area(area_km2, sensor=sensor)

    return (round(area_km2, 4), is_valid, recommended_resolution_m)


def validate_aoi_geometry(
    geometry_4326: dict[str, object] | BaseGeometry,
    max_area_km2: float | None = None,
    sensor: str | None = None,
) -> tuple[float, bool, int]:
    """Compute exact area in km^2 in the local metric CRS (D96/TM in SLO / local UTM globally)."""
    geom_4326 = geometry_4326 if isinstance(geometry_4326, BaseGeometry) else shape(geometry_4326)
    if geom_4326.is_empty:
        return (0.0, False, 30 if (sensor and str(sensor).strip().lower() == "landsat-c2-l2") else 10)

    eff_max_area = (
        float(max_area_km2)
        if max_area_km2 is not None
        else float(get_runtime_environment()["max_area_km2"])
    )

    min_x, min_y, max_x, max_y = geom_4326.bounds
    max_seg = max((max_x - min_x) / 16.0, (max_y - min_y) / 16.0, 1e-4)
    geom_dense = geom_4326.segmentize(max_segment_length=max_seg)

    c_lat = 0.5 * (min_y + max_y)
    c_lon = 0.5 * (min_x + max_x)
    metric_crs, _ = get_best_metric_crs(c_lat, c_lon, mode="auto")
    transformer = _get_transformer(CRS_WGS84, metric_crs)
    geom_metric = shapely_transform(transformer.transform, geom_dense)
    area_km2 = float(geom_metric.area / 1_000_000.0)
    is_valid = (area_km2 > 0.0) and (area_km2 <= eff_max_area * 1.02)
    recommended_resolution_m = _recommended_resolution_for_area(area_km2, sensor=sensor)

    return (round(area_km2, 4), is_valid, recommended_resolution_m)


def clamp_polygon_to_max_area(
    polygon_geojson_4326: dict[str, object] | BaseGeometry,
    max_area_km2: float | None = None,
) -> tuple[dict[str, object], bool, float]:
    """Scale an oversized EPSG:4326 polygon symmetrically around its metric centroid to fit ``max_area_km2``.

    Returns ``(geojson_dict, was_clamped, new_area_km2)`` so both the polygon and its enclosing
    WGS84 bbox are guaranteed to stay within RAM limits.
    """
    from shapely.affinity import scale as shapely_scale

    geom_4326 = (
        polygon_geojson_4326
        if isinstance(polygon_geojson_4326, BaseGeometry)
        else shape(polygon_geojson_4326)
    )
    if not geom_4326.is_valid:
        geom_4326 = geom_4326.buffer(0)
    if geom_4326.is_empty:
        return dict(mapping(geom_4326)), False, 0.0

    eff_max = (
        float(max_area_km2)
        if max_area_km2 is not None
        else float(get_ram_limits()["max_area_km2"])
    )
    poly_area, poly_ok, _ = validate_aoi_geometry(geom_4326, max_area_km2=eff_max)
    min_x, min_y, max_x, max_y = geom_4326.bounds
    bbox_area, bbox_ok, _ = validate_aoi_size([min_x, min_y, max_x, max_y], max_area_km2=eff_max)

    if poly_ok and bbox_ok:
        return dict(mapping(geom_4326)), False, poly_area

    # Scale symmetrically in local metric CRS around centroid so enclosing bbox fits 0.96 * eff_max
    c_lat = 0.5 * (min_y + max_y)
    c_lon = 0.5 * (min_x + max_x)
    metric_crs, _ = get_best_metric_crs(c_lat, c_lon, mode="auto")
    fwd = _get_transformer(CRS_WGS84, metric_crs)
    inv = _get_transformer(metric_crs, CRS_WGS84)

    geom_m = shapely_transform(fwd.transform, geom_4326)
    worst_area = max(poly_area, bbox_area, 1e-6)
    factor = float(np.sqrt((eff_max * 0.95) / worst_area))
    factor = min(max(factor, 0.05), 0.99)

    clamped_4326 = geom_4326
    for _ in range(3):
        scaled_m = shapely_scale(geom_m, xfact=factor, yfact=factor, origin="centroid")
        clamped_4326 = shapely_transform(inv.transform, scaled_m)
        b_x0, b_y0, b_x1, b_y1 = clamped_4326.bounds
        b_area, b_ok, _ = validate_aoi_size([b_x0, b_y0, b_x1, b_y1], max_area_km2=eff_max)
        if b_ok and b_area <= eff_max * 1.01:
            break
        factor *= 0.92

    new_area, _, _ = validate_aoi_geometry(clamped_4326, max_area_km2=eff_max)
    return dict(mapping(clamped_4326)), True, new_area


def clip_to_aoi_polygon(
    obj: xr.DataArray | xr.Dataset,
    polygon_geojson_4326: dict[str, object] | BaseGeometry | None,
    drop: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Clip an xarray DataArray/Dataset to an exact EPSG:4326 polygon using rioxarray.

    Pixels outside the drawn/uploaded polygon are masked to each variable's nodata
    value (``np.nan`` for float32 reflectance/indices/SAR dB, ``0`` for SCL/QA),
    so all area statistics and time series reflect strictly the exact polygon.
    """
    if polygon_geojson_4326 is None:
        return obj

    geom_dict = (
        mapping(polygon_geojson_4326)
        if isinstance(polygon_geojson_4326, BaseGeometry)
        else polygon_geojson_4326
    )

    obj = _ensure_spatial_metadata(obj)
    src_crs = obj.rio.crs

    try:
        clipped = obj.rio.clip(
            [geom_dict],
            crs=CRS_WGS84,
            drop=drop,
            all_touched=False,
        )
    except NoDataInBounds:
        clipped = obj.rio.clip(
            [geom_dict],
            crs=CRS_WGS84,
            drop=drop,
            all_touched=True,
        )

    if src_crs is not None:
        clipped = clipped.rio.write_crs(src_crs, inplace=False)
    return clipped


def _ensure_spatial_metadata(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Ensure xarray object has recognized spatial dims and a valid CRS."""
    dims = set(obj.dims)
    if "x" in dims and "y" in dims:
        obj = obj.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    elif "longitude" in dims and "latitude" in dims:
        obj = obj.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)

    if obj.rio.crs is None:
        raise ValueError(
            "Input xarray object has no CRS attached. "
            "Set a CRS with `.rio.write_crs(...)` before exporting to GeoTIFF."
        )
    return obj


def export_to_geotiff_bytes(
    data_array: xr.DataArray | xr.Dataset,
    target_epsg: int | str | CRS = EPSG_D96TM,
) -> bytes:
    """Reproject an xarray DataArray/Dataset and serialize to LZW-compressed GeoTIFF bytes.

    Designed for direct use with ``streamlit.download_button`` without writing
    temporary files to disk.

    Parameters
    ----------
    data_array : xr.DataArray | xr.Dataset
        Georeferenced 2D ``(y, x)`` or 3D ``(band, y, x)`` DataArray, or a Dataset
        of 2D ``(y, x)`` bands. A singleton ``time`` dimension is squeezed automatically.
    target_epsg : int | str | pyproj.CRS, default 3794
        Target CRS for export (e.g., ``3794`` for D96/TM or ``32633`` for UTM 33N).

    Returns
    -------
    bytes
        In-memory GeoTIFF byte stream with LZW compression, CRS, and nodata tags.
    """
    if not isinstance(data_array, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"Expected xarray.DataArray or xarray.Dataset, got {type(data_array).__name__}."
        )

    target_crs = normalize_crs(target_epsg)
    obj: xr.DataArray | xr.Dataset = data_array

    # Squeeze singleton time dimension if present
    if "time" in obj.dims:
        if obj.sizes["time"] == 1:
            obj = obj.isel(time=0, drop=True)
        else:
            raise ValueError(
                f"GeoTIFF export requires a single time slice, got time size={obj.sizes['time']}."
            )

    obj = _ensure_spatial_metadata(obj)
    src_crs = CRS.from_user_input(obj.rio.crs)

    band_names: list[str] = []
    if isinstance(obj, xr.Dataset):
        if not obj.data_vars:
            raise ValueError("Cannot export an empty xarray.Dataset to GeoTIFF.")
        band_names = [str(name) for name in obj.data_vars]
        # Stack variables into a single (band, y, x) DataArray while keeping float32 lean
        common_dtype = np.result_type(*(obj[name].dtype for name in obj.data_vars))
        if np.issubdtype(common_dtype, np.floating) and common_dtype == np.float64:
            common_dtype = np.float32
        da = obj.to_array(dim="band").astype(common_dtype, copy=False)
        da = da.rio.write_crs(src_crs, inplace=False)
        da = da.rio.set_spatial_dims(x_dim=obj.rio.x_dim, y_dim=obj.rio.y_dim, inplace=False)
    else:
        da = obj
        if da.ndim == 2 and da.name is not None:
            band_names = [str(da.name)]
        elif da.ndim == 3 and "band" in da.dims and "band" in da.coords:
            band_names = [str(v) for v in da.coords["band"].values]

    # Ensure valid nodata tag for the output dtype
    is_float = np.issubdtype(da.dtype, np.floating)
    nodata = da.rio.nodata
    if nodata is None:
        nodata = np.nan if is_float else 0
    da = da.rio.write_nodata(nodata, inplace=False)

    # Reproject only if target CRS differs from current CRS
    if src_crs != target_crs:
        resampling = Resampling.bilinear if is_float else Resampling.nearest
        da_export = da.rio.reproject(target_crs, resampling=resampling, nodata=nodata)
    else:
        da_export = da

    # Write LZW-compressed GeoTIFF into an in-memory GDAL virtual file
    with MemoryFile() as memfile:
        da_export.rio.to_raster(
            memfile.name,
            driver="GTiff",
            compress="LZW",
            BIGTIFF="IF_SAFER",
        )
        if band_names:
            with rasterio.open(memfile.name, "r+") as dst:
                if dst.count == len(band_names):
                    for idx, name in enumerate(band_names, start=1):
                        dst.set_band_description(idx, name)
        memfile.seek(0)
        tif_bytes = memfile.read()

    del da_export
    if isinstance(obj, xr.Dataset):
        del da
    gc.collect()

    return tif_bytes
