"""Copernicus DEM 30 m terrain analysis & metric cross-section profile engine.

Streams ``cop-dem-glo-30`` from Microsoft Planetary Computer STAC directly into the
active metric CRS (``EPSG:3794`` D96/TM in Slovenia or local auto-UTM worldwide)
and computes:
- Elevation [m a.s.l.]
- Slope [°] and Slope [%] from exact metric gradients ``(dx, dy)``
- Aspect [0–360°] compass azimuth of steepest descent (``-1`` for flat areas)
- Analytical Hillshade [0–255] (``sun_azimuth=315°``, ``sun_altitude=45°``)
- Terrain morphometry statistics (min/mean/max height, relief energy, steepness areas)
- 150-point metric cross-section profile (``A -> B``) with optional optical/thermal
  index overlay (NDVI, NDSI, LST °C, etc.) along the exact same geodetic transect.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Sequence

import numpy as np
import odc.stac
import pandas as pd
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr

from geodezija_crs import (
    CRS_D96TM,
    clip_to_aoi_polygon,
    get_best_metric_crs,
    normalize_crs,
    transform_coords,
)
from stac_loader import get_stac_catalog

logger = logging.getLogger(__name__)

TERRAIN_LAYER_SPECS: dict[str, dict[str, Any]] = {
    "elevation": {
        "cmap": "terrain",
        "units": "m n.m.",
        "auto_range": True,
        "vmin": 0.0,
        "vmax": 2500.0,
    },
    "slope_deg": {
        "cmap": "YlOrRd",
        "units": "°",
        "auto_range": False,
        "vmin": 0.0,
        "vmax": 45.0,
    },
    "aspect": {
        "cmap": "twilight",
        "units": "°",
        "auto_range": False,
        "vmin": 0.0,
        "vmax": 360.0,
    },
    "hillshade": {
        "cmap": "Greys_r",
        "units": "",
        "auto_range": False,
        "vmin": 0.0,
        "vmax": 255.0,
    },
}


def load_copernicus_dem(
    bbox_4326: Sequence[float],
    target_epsg: int | str = CRS_D96TM,
    resolution: float = 30.0,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> xr.DataArray:
    """Stream & mosaic Copernicus DEM 30 m (``cop-dem-glo-30``) in the active metric CRS.

    Parameters
    ----------
    bbox_4326 : Sequence[float]
        AOI bounding box ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
    target_epsg : int | str, default ``"EPSG:3794"``
        Target projected metric CRS (``"EPSG:3794"`` or local UTM ``"EPSG:326xx/327xx"``).
        Passing ``"auto"`` resolves the exact local metric CRS from the AOI center.
    resolution : float, default 30.0
        Output pixel spacing in meters (native Copernicus DEM GLO-30 is ~30 m).
    aoi_polygon_4326 : dict | None
        Optional GeoJSON polygon in EPSG:4326 for exact ``rio.clip`` masking.

    Returns
    -------
    xr.DataArray
        2D ``float32`` georeferenced elevation raster [m a.s.l.] with ``np.nan`` nodata.
    """
    if len(bbox_4326) != 4:
        raise ValueError(f"bbox_4326 must have 4 values, got {len(bbox_4326)}.")
    if float(resolution) <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    if not target_epsg or str(target_epsg).strip().lower() == "auto":
        c_lon = 0.5 * (float(bbox_4326[0]) + float(bbox_4326[2]))
        c_lat = 0.5 * (float(bbox_4326[1]) + float(bbox_4326[3]))
        crs_str, _ = get_best_metric_crs(c_lat, c_lon, mode="auto")
    else:
        crs_obj = normalize_crs(target_epsg)
        epsg_num = crs_obj.to_epsg()
        crs_str = f"EPSG:{epsg_num}" if epsg_num is not None else crs_obj.to_wkt()

    catalog = get_stac_catalog()
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[float(v) for v in bbox_4326],
        max_items=12,
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Copernicus DEM 30 m (cop-dem-glo-30) tiles intersect the selected AOI.")

    ds = odc.stac.load(
        items,
        bands=["data"],
        bbox=[float(v) for v in bbox_4326],
        crs=crs_str,
        resolution=float(resolution),
        resampling="bilinear",
        groupby="solar_day",
    )
    if "time" in ds.dims:
        da = ds["data"].isel(time=0, drop=True).compute().astype(np.float32)
    else:
        da = ds["data"].compute().astype(np.float32)

    da = da.where(np.isfinite(da) & (da > -500.0))
    da = da.rio.write_crs(crs_str, inplace=False)
    da = da.rio.write_nodata(np.nan, inplace=False)

    if aoi_polygon_4326 is not None:
        da = clip_to_aoi_polygon(da, aoi_polygon_4326, drop=False)

    da = da.assign_attrs(
        {
            "long_name": "Copernicus DEM 30 m Elevation",
            "units": "m",
            "source": "cop-dem-glo-30",
            "resolution_m": float(resolution),
            "crs": crs_str,
            "tile_count": len(items),
        }
    )
    del ds
    gc.collect()
    return da


def compute_terrain_derivatives(
    dem_da: xr.DataArray,
    *,
    sun_azimuth_deg: float = 315.0,
    sun_altitude_deg: float = 45.0,
    flat_slope_thresh_deg: float = 0.5,
) -> dict[str, Any]:
    """Compute exact metric Slope [° & %], Aspect [0–360°], Hillshade [0–255], and summary stats.

    Uses exact metric pixel spacing ``(dx, dy)`` extracted from ``dem_da`` coordinates
    and respects north-up vs south-up row ordering so aspect & hillshade illumination
    azimuths are 100% geodetically accurate.
    """
    if dem_da.ndim != 2:
        raise ValueError(f"Expected 2D DEM DataArray (y, x), got shape {dem_da.shape}.")

    z = np.asarray(dem_da.values, dtype=np.float64)
    x_vals = np.asarray(dem_da.coords["x"].values, dtype=np.float64)
    y_vals = np.asarray(dem_da.coords["y"].values, dtype=np.float64)

    if x_vals.size < 2 or y_vals.size < 2:
        raise ValueError("DEM raster must have at least 2x2 pixels to compute terrain gradients.")

    # Signed spacing so dz/dx is strictly East-ward and dz/dy is strictly North-ward
    dx_signed = float(np.median(np.diff(x_vals)))
    dy_signed = float(np.median(np.diff(y_vals)))
    if dx_signed == 0.0 or dy_signed == 0.0:
        raise ValueError(f"Invalid DEM coordinate spacing: dx={dx_signed}, dy={dy_signed}.")

    dx_abs = abs(dx_signed)
    dy_abs = abs(dy_signed)
    valid = np.isfinite(z)

    # Fill isolated interior/border NaNs temporarily with local median solely so np.gradient
    # does not widen NaN borders unnecessarily, then strictly mask back to `valid`.
    z_work = z.copy()
    if np.any(valid) and not np.all(valid):
        z_work[~valid] = float(np.nanmedian(z))

    dz_dy_raw, dz_dx_raw = np.gradient(z_work, dy_signed, dx_signed)
    # np.gradient divides by signed spacing:
    # - if y decreases downward (standard north-up raster), dy_signed < 0,
    #   so dz_dy_raw = dZ / d(Northing) pointing strictly North!
    # - if x increases rightward, dx_signed > 0, so dz_dx_raw = dZ / d(Easting) pointing strictly East!
    dz_dx = dz_dx_raw
    dz_dy = dz_dy_raw

    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_rad = np.arctan(grad_mag)
    slope_deg = np.degrees(slope_rad)
    slope_pct = np.tan(slope_rad) * 100.0

    # Compass azimuth of steepest descent vector (-dz_dx, -dz_dy):
    # 0° = North, 90° = East, 180° = South, 270° = West; -1.0 for flat terrain
    aspect_deg = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360.0) % 360.0
    is_flat = slope_deg < float(flat_slope_thresh_deg)
    aspect_out = np.where(is_flat, -1.0, aspect_deg)

    # Analytical hillshade [0..255] with azimuth clockwise from North & altitude above horizon
    zenith_rad = np.radians(90.0 - float(sun_altitude_deg))
    sun_az_rad = np.radians(float(sun_azimuth_deg))
    aspect_rad = np.radians(np.where(is_flat, 0.0, aspect_deg))
    hs_norm = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(sun_az_rad - aspect_rad)
    )
    hillshade = np.clip(hs_norm, 0.0, 1.0) * 255.0

    # Strictly re-apply original valid DEM mask
    elev_f32 = np.where(valid, z, np.nan).astype(np.float32)
    slope_deg_f32 = np.where(valid, slope_deg, np.nan).astype(np.float32)
    slope_pct_f32 = np.where(valid, slope_pct, np.nan).astype(np.float32)
    aspect_f32 = np.where(valid, aspect_out, np.nan).astype(np.float32)
    hillshade_f32 = np.where(valid, hillshade, np.nan).astype(np.float32)

    src_crs = dem_da.rio.crs
    crs_str = str(src_crs) if src_crs is not None else str(dem_da.attrs.get("crs", CRS_D96TM))
    res_m = round(0.5 * (dx_abs + dy_abs), 2)

    def _wrap(arr: np.ndarray, name: str, long_name: str, units: str) -> xr.DataArray:
        da_out = xr.DataArray(
            arr,
            dims=dem_da.dims,
            coords=dem_da.coords,
            name=name,
            attrs={
                "long_name": long_name,
                "units": units,
                "source": "cop-dem-glo-30",
                "resolution_m": res_m,
                "crs": crs_str,
            },
        )
        if src_crs is not None:
            da_out = da_out.rio.write_crs(src_crs, inplace=False)
        da_out = da_out.rio.write_nodata(np.nan, inplace=False)
        return da_out

    elevation_da = _wrap(elev_f32, "DEM_ELEVATION", "Elevation [m a.s.l.]", "m")
    slope_deg_da = _wrap(slope_deg_f32, "DEM_SLOPE_DEG", "Terrain Slope [°]", "deg")
    slope_pct_da = _wrap(slope_pct_f32, "DEM_SLOPE_PCT", "Terrain Slope [%]", "%")
    aspect_da = _wrap(aspect_f32, "DEM_ASPECT_DEG", "Terrain Aspect [0-360°, -1=Flat]", "deg")
    hillshade_da = _wrap(
        hillshade_f32,
        "DEM_HILLSHADE",
        f"Analytical Hillshade (az={sun_azimuth_deg:.0f}°, alt={sun_altitude_deg:.0f}°)",
        "0-255",
    )

    valid_count = int(np.count_nonzero(valid))
    px_area_ha = (dx_abs * dy_abs) / 10_000.0
    valid_area_ha = valid_count * px_area_ha

    if valid_count > 0:
        z_v = elev_f32[valid]
        s_v = slope_deg_f32[valid]
        sp_v = slope_pct_f32[valid]
        steep_20_px = int(np.count_nonzero(s_v > 20.0))
        steep_30_px = int(np.count_nonzero(s_v > 30.0))
        stats: dict[str, Any] = {
            "min_h": round(float(np.min(z_v)), 1),
            "mean_h": round(float(np.mean(z_v)), 1),
            "median_h": round(float(np.median(z_v)), 1),
            "max_h": round(float(np.max(z_v)), 1),
            "relief_diff_m": round(float(np.max(z_v) - np.min(z_v)), 1),
            "mean_slope_deg": round(float(np.mean(s_v)), 2),
            "max_slope_deg": round(float(np.max(s_v)), 2),
            "mean_slope_pct": round(float(np.mean(sp_v)), 1),
            "steep_20_ha": round(steep_20_px * px_area_ha, 2),
            "steep_20_pct": round(100.0 * steep_20_px / valid_count, 1),
            "steep_30_ha": round(steep_30_px * px_area_ha, 2),
            "steep_30_pct": round(100.0 * steep_30_px / valid_count, 1),
            "valid_pixels": valid_count,
            "valid_area_ha": round(valid_area_ha, 2),
            "resolution_m": res_m,
            "crs": crs_str,
        }
    else:
        stats = {
            "min_h": float("nan"),
            "mean_h": float("nan"),
            "median_h": float("nan"),
            "max_h": float("nan"),
            "relief_diff_m": 0.0,
            "mean_slope_deg": 0.0,
            "max_slope_deg": 0.0,
            "mean_slope_pct": 0.0,
            "steep_20_ha": 0.0,
            "steep_20_pct": 0.0,
            "steep_30_ha": 0.0,
            "steep_30_pct": 0.0,
            "valid_pixels": 0,
            "valid_area_ha": 0.0,
            "resolution_m": res_m,
            "crs": crs_str,
        }

    return {
        "elevation": elevation_da,
        "slope_deg": slope_deg_da,
        "slope_pct": slope_pct_da,
        "aspect": aspect_da,
        "hillshade": hillshade_da,
        "stats": stats,
    }


def build_preset_transect_endpoints(
    dem_da: xr.DataArray,
    preset: str = "w_e",
    margin_frac: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return inset metric endpoints ``((x_a, y_a), (x_b, y_b))`` for standard AOI transects.

    Supported presets:
    - ``"w_e"``: West -> East across AOI center
    - ``"s_n"``: South -> North across AOI center
    - ``"sw_ne"``: South-West -> North-East diagonal
    """
    x_vals = np.asarray(dem_da.coords["x"].values, dtype=np.float64)
    y_vals = np.asarray(dem_da.coords["y"].values, dtype=np.float64)
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    dx = x_max - x_min
    dy = y_max - y_min
    mx = dx * float(margin_frac)
    my = dy * float(margin_frac)
    xc = 0.5 * (x_min + x_max)
    yc = 0.5 * (y_min + y_max)

    p = str(preset or "w_e").strip().lower()
    if p == "s_n":
        return (xc, y_min + my), (xc, y_max - my)
    if p == "sw_ne":
        return (x_min + mx, y_min + my), (x_max - mx, y_max - my)
    return (x_min + mx, yc), (x_max - mx, yc)


def _sample_raster_along_coords(
    da: xr.DataArray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Sample a 2D georeferenced DataArray bilinearly at metric ``(xs, ys)`` using pure NumPy."""
    arr = np.asarray(da.values, dtype=np.float64)
    x_coords = np.asarray(da.coords["x"].values, dtype=np.float64)
    y_coords = np.asarray(da.coords["y"].values, dtype=np.float64)
    h, w = arr.shape
    out = np.full(xs.shape, np.nan, dtype=np.float64)
    if h == 0 or w == 0:
        return out

    # Normalize to strictly ascending coordinate axes for searchsorted
    if x_coords.size > 1 and x_coords[0] > x_coords[-1]:
        x_coords = x_coords[::-1]
        arr = arr[:, ::-1]
    if y_coords.size > 1 and y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        arr = arr[::-1, :]

    in_bounds = (
        (xs >= x_coords[0])
        & (xs <= x_coords[-1])
        & (ys >= y_coords[0])
        & (ys <= y_coords[-1])
    )
    if not np.any(in_bounds):
        return out

    xq = xs[in_bounds]
    yq = ys[in_bounds]

    ix1 = np.clip(np.searchsorted(x_coords, xq, side="right"), 1, max(1, w - 1))
    ix0 = np.clip(ix1 - 1, 0, max(0, w - 1))
    iy1 = np.clip(np.searchsorted(y_coords, yq, side="right"), 1, max(1, h - 1))
    iy0 = np.clip(iy1 - 1, 0, max(0, h - 1))

    dx = x_coords[ix1] - x_coords[ix0]
    dy = y_coords[iy1] - y_coords[iy0]
    tx = np.where(dx != 0.0, (xq - x_coords[ix0]) / np.where(dx != 0.0, dx, 1.0), 0.0)
    ty = np.where(dy != 0.0, (yq - y_coords[iy0]) / np.where(dy != 0.0, dy, 1.0), 0.0)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    v00 = arr[iy0, ix0]
    v10 = arr[iy0, ix1]
    v01 = arr[iy1, ix0]
    v11 = arr[iy1, ix1]

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    # Weighted bilinear average ignoring NaN corners when at least one neighbor is finite
    num = np.zeros_like(xq, dtype=np.float64)
    den = np.zeros_like(xq, dtype=np.float64)
    for v_c, w_c in ((v00, w00), (v10, w10), (v01, w01), (v11, w11)):
        fin = np.isfinite(v_c)
        num += np.where(fin, v_c * w_c, 0.0)
        den += np.where(fin, w_c, 0.0)

    vals = np.full_like(xq, np.nan, dtype=np.float64)
    ok = den > 0.0
    vals[ok] = num[ok] / den[ok]
    out[in_bounds] = vals
    return out


def extract_transect_profile(
    dem_da: xr.DataArray,
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    *,
    n_samples: int = 150,
    overlay_da: xr.DataArray | None = None,
    overlay_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample 150 evenly spaced points along exact metric line ``A -> B`` on DEM (+ optional index).

    Parameters
    ----------
    dem_da : xr.DataArray
        2D metric elevation DataArray [m a.s.l.].
    point_a, point_b : tuple[float, float]
        Transect endpoints ``(E, N)`` in ``dem_da``'s metric CRS.
    n_samples : int, default 150
        Number of evenly spaced profile sample points along ``A -> B``.
    overlay_da : xr.DataArray | None
        Optional 2D optical/thermal raster (e.g., ``NDVI``, ``NDSI``, ``LST``) to sample
        along the exact same geographic line.
    overlay_name : str | None
        Display name for the overlay column/series (e.g. ``"NDVI"``, ``"LST [°C]"``).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        ``(profile_df, summary)`` with ``distance_m``, ``distance_km``, ``x``, ``y``,
        ``elevation_m``, and optional ``overlay_value``.
    """
    n_pts = max(2, int(n_samples))
    xa, ya = float(point_a[0]), float(point_a[1])
    xb, yb = float(point_b[0]), float(point_b[1])

    total_length_m = float(np.hypot(xb - xa, yb - ya))
    if total_length_m <= 0.0:
        raise ValueError("Transect endpoints Point A and Point B must be distinct.")

    t_vals = np.linspace(0.0, 1.0, n_pts, dtype=np.float64)
    xs = xa + t_vals * (xb - xa)
    ys = ya + t_vals * (yb - ya)
    dists_m = t_vals * total_length_m

    elev_vals = _sample_raster_along_coords(dem_da, xs, ys)

    overlay_vals: np.ndarray | None = None
    has_overlay = False
    eff_overlay_name = str(overlay_name or "").strip()

    if overlay_da is not None and overlay_da.ndim == 2:
        try:
            dem_crs = dem_da.rio.crs
            ov_crs = overlay_da.rio.crs
            if dem_crs is not None and ov_crs is not None and normalize_crs(dem_crs) != normalize_crs(ov_crs):
                ov_xs = np.empty_like(xs)
                ov_ys = np.empty_like(ys)
                for idx_pt in range(n_pts):
                    ox, oy = transform_coords(xs[idx_pt], ys[idx_pt], dem_crs, ov_crs)
                    ov_xs[idx_pt] = ox
                    ov_ys[idx_pt] = oy
            else:
                ov_xs, ov_ys = xs, ys
            overlay_vals = _sample_raster_along_coords(overlay_da, ov_xs, ov_ys)
            has_overlay = bool(np.any(np.isfinite(overlay_vals)))
            if not eff_overlay_name:
                eff_overlay_name = str(overlay_da.name or overlay_da.attrs.get("index_name", "Overlay"))
        except Exception as exc:
            logger.warning("Failed sampling overlay along transect: %s", exc)
            overlay_vals = None
            has_overlay = False

    data: dict[str, Any] = {
        "distance_m": np.round(dists_m, 2),
        "distance_km": np.round(dists_m / 1000.0, 4),
        "x_m": np.round(xs, 2),
        "y_m": np.round(ys, 2),
        "elevation_m": np.round(elev_vals, 2),
    }
    if has_overlay and overlay_vals is not None:
        data["overlay_value"] = np.round(overlay_vals, 4)

    profile_df = pd.DataFrame(data)

    fin_mask = np.isfinite(elev_vals)
    fin_elev = elev_vals[fin_mask]
    fin_dist = dists_m[fin_mask]

    if fin_elev.size >= 2:
        dh = np.diff(fin_elev)
        dd = np.diff(fin_dist)
        valid_steps = dd > 0.0
        total_ascent_m = float(np.sum(np.maximum(dh, 0.0)))
        total_descent_m = float(np.sum(np.maximum(-dh, 0.0)))
        mean_slope_pct = (
            float(100.0 * np.mean(np.abs(dh[valid_steps]) / dd[valid_steps]))
            if np.any(valid_steps)
            else 0.0
        )
        min_h = float(np.min(fin_elev))
        max_h = float(np.max(fin_elev))
        net_gain_m = float(fin_elev[-1] - fin_elev[0])
    elif fin_elev.size == 1:
        total_ascent_m = 0.0
        total_descent_m = 0.0
        mean_slope_pct = 0.0
        min_h = float(fin_elev[0])
        max_h = float(fin_elev[0])
        net_gain_m = 0.0
    else:
        total_ascent_m = 0.0
        total_descent_m = 0.0
        mean_slope_pct = 0.0
        min_h = float("nan")
        max_h = float("nan")
        net_gain_m = 0.0

    summary: dict[str, Any] = {
        "point_a": (round(xa, 2), round(ya, 2)),
        "point_b": (round(xb, 2), round(yb, 2)),
        "n_samples": n_pts,
        "total_length_m": round(total_length_m, 1),
        "total_length_km": round(total_length_m / 1000.0, 3),
        "total_ascent_m": round(total_ascent_m, 1),
        "total_descent_m": round(total_descent_m, 1),
        "net_gain_m": round(net_gain_m, 1),
        "min_h": round(min_h, 1) if np.isfinite(min_h) else float("nan"),
        "max_h": round(max_h, 1) if np.isfinite(max_h) else float("nan"),
        "mean_slope_pct": round(mean_slope_pct, 1),
        "has_overlay": has_overlay,
        "overlay_name": eff_overlay_name if has_overlay else None,
    }
    return profile_df, summary
