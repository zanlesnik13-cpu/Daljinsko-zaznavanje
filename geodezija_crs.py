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

CRS_D96TM: str = "EPSG:3794"
CRS_WGS84: str = "EPSG:4326"
CRS_UTM33N: str = "EPSG:32633"

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
}

# RAM-safe resolution step-down thresholds (km^2) for 1 GB Streamlit Cloud limit
_RES_10M_MAX_KM2: float = 100.0
_RES_20M_MAX_KM2: float = 350.0
_RES_30M_MAX_KM2: float = 750.0


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


def _d96tm_box_to_wgs84_bbox(
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
) -> list[float]:
    """Convert an EPSG:3794 bounding box to its enclosing EPSG:4326 bounding box.

    Edges are densified before reprojection so meridian convergence and
    projection curvature are accurately captured.
    """
    geom_3794 = box(min_e, min_n, max_e, max_n)
    max_seg = max((max_e - min_e) / 16.0, (max_n - min_n) / 16.0, 50.0)
    geom_3794 = geom_3794.segmentize(max_segment_length=max_seg)

    transformer = _get_transformer(CRS_D96TM, CRS_WGS84)
    geom_4326 = shapely_transform(transformer.transform, geom_3794)
    min_lon, min_lat, max_lon, max_lat = geom_4326.bounds
    return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


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
    """Build an exact square AOI around a WGS84 point ``(lat, lon)``.

    Projects ``(lon, lat)`` to EPSG:3794 (D96/TM), buffers by exact metric
    ``radius_m``, and returns both the enclosing EPSG:4326 bounding box and the
    exact EPSG:3794 metric bounding box.

    Parameters
    ----------
    lat : float
        Latitude in degrees (EPSG:4326), in ``[-90, 90]``.
    lon : float
        Longitude in degrees (EPSG:4326), in ``[-180, 180]``.
    radius_m : float
        Half-width of the square AOI in meters (total side length = ``2 * radius_m``).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(bbox_4326, bbox_3794)`` where:
        - ``bbox_4326 = [min_lon, min_lat, max_lon, max_lat]``
        - ``bbox_3794 = [min_e, min_n, max_e, max_n]``
    """
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}.")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}.")
    if radius_m <= 0:
        raise ValueError(f"radius_m must be strictly positive, got {radius_m}.")

    e, n = transform_coords(lon, lat, EPSG_WGS84, EPSG_D96TM)
    return bbox_from_point_d96tm(e, n, radius_m)


def _recommended_resolution_for_area(area_km2: float) -> int:
    """Return a RAM-safe target spatial resolution (10 / 20 / 30 / 60 m) for AOI area."""
    if area_km2 <= _RES_10M_MAX_KM2:
        return 10
    if area_km2 <= _RES_20M_MAX_KM2:
        return 20
    if area_km2 <= _RES_30M_MAX_KM2:
        return 30
    return 60


def validate_aoi_size(
    bbox_4326: Sequence[float],
    max_area_km2: float = 900.0,
) -> tuple[float, bool, int]:
    """Validate an EPSG:4326 AOI bounding box by computing exact area in EPSG:3794.

    Also recommends a RAM-safe target resolution (10 m -> 20 m -> 30 m -> 60 m)
    to stay comfortably within Streamlit Community Cloud's 1 GB RAM limit.

    Parameters
    ----------
    bbox_4326 : Sequence[float]
        Bounding box ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    max_area_km2 : float, default 900.0
        Maximum permitted AOI area in square kilometers.

    Returns
    -------
    tuple[float, bool, int]
        ``(area_km2, is_valid, recommended_resolution_m)``
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

    geom_4326 = box(min_lon, min_lat, max_lon, max_lat)
    max_seg = max((max_lon - min_lon) / 16.0, (max_lat - min_lat) / 16.0, 1e-4)
    geom_4326 = geom_4326.segmentize(max_segment_length=max_seg)

    transformer = _get_transformer(CRS_WGS84, CRS_D96TM)
    geom_3794 = shapely_transform(transformer.transform, geom_4326)
    area_km2 = float(geom_3794.area / 1_000_000.0)

    is_valid = (area_km2 > 0.0) and (area_km2 <= float(max_area_km2))
    recommended_resolution_m = _recommended_resolution_for_area(area_km2)

    return (round(area_km2, 4), is_valid, recommended_resolution_m)


def validate_aoi_geometry(
    geometry_4326: dict[str, object] | BaseGeometry,
    max_area_km2: float = 900.0,
) -> tuple[float, bool, int]:
    """Compute exact area in km^2 in EPSG:3794 for an arbitrary EPSG:4326 polygon.

    Returns ``(area_km2, is_valid, recommended_resolution_m)`` using the same
    1 GB RAM-safe resolution tiers as :func:`validate_aoi_size`.
    """
    geom_4326 = geometry_4326 if isinstance(geometry_4326, BaseGeometry) else shape(geometry_4326)
    if geom_4326.is_empty:
        return (0.0, False, 10)

    min_x, min_y, max_x, max_y = geom_4326.bounds
    max_seg = max((max_x - min_x) / 16.0, (max_y - min_y) / 16.0, 1e-4)
    geom_dense = geom_4326.segmentize(max_segment_length=max_seg)

    transformer = _get_transformer(CRS_WGS84, CRS_D96TM)
    geom_3794 = shapely_transform(transformer.transform, geom_dense)
    area_km2 = float(geom_3794.area / 1_000_000.0)
    is_valid = (area_km2 > 0.0) and (area_km2 <= float(max_area_km2))
    recommended_resolution_m = _recommended_resolution_for_area(area_km2)

    return (round(area_km2, 4), is_valid, recommended_resolution_m)


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
