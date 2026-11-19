"""Memory-safe multi-sensor time-series extraction and pixel-aligned change detection.

Supports Sentinel-2 L2A, Landsat C2 L2 (1984–present), and Sentinel-1 SAR RTC,
processing strictly one epoch in RAM at a time (Streamlit Cloud 1 GB RAM safe)
and supporting exact polygon clipping (`rioxarray.rio.clip`).
"""

from __future__ import annotations

import gc
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pystac
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr
from rasterio.enums import Resampling

from analize import (
    apply_cloud_mask,
    compute_index,
    compute_index_stats,
    extract_sar_metric,
    get_required_bands_for_sensor,
)
from stac_loader import SENSOR_S1, SENSOR_S2, load_scene

logger = logging.getLogger(__name__)

TIME_SERIES_COLUMNS: list[str] = [
    "date",
    "date_str",
    "scene_id",
    "platform",
    "mgrs_tile",
    "cloud_cover",
    "mean",
    "median",
    "std",
    "p10",
    "p90",
    "min",
    "max",
    "valid_pct",
    "valid_pixels",
    "valid_area_ha",
]


def _item_dt_utc(item: pystac.Item) -> datetime:
    dt = item.datetime
    if dt is None:
        raw = item.properties.get("datetime")
        if raw is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _compute_layer(ds: xr.Dataset, sensor: str, metric_name: str) -> xr.DataArray:
    """Compute either an optical spectral index or a SAR backscatter metric."""
    if sensor == SENSOR_S1:
        return extract_sar_metric(ds, metric_name)
    return compute_index(ds, metric_name)


def extract_time_series(
    items: Sequence[pystac.Item],
    bbox_4326: Sequence[float],
    index_name: str,
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    mask_clouds: bool = True,
    min_valid_pct: float = 10.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract a multi-temporal index or SAR backscatter time series (1 epoch in RAM).

    Streams only the minimal bands needed for ``sensor`` and ``index_name``,
    clips to ``aoi_polygon_4326`` if provided, applies sensor-appropriate cloud
    masking (Sentinel-2 SCL or Landsat QA_PIXEL; skipped for all-weather SAR),
    records statistics, and immediately releases each raster via ``gc.collect()``.
    """
    idx_key = index_name.strip().upper()
    use_cloud_mask = mask_clouds and (sensor != SENSOR_S1)
    req_bands = get_required_bands_for_sensor(sensor, idx_key, include_cloud_mask=use_cloud_mask)

    sorted_items = sorted(items, key=_item_dt_utc)
    total = len(sorted_items)
    rows: list[dict[str, Any]] = []
    skipped_epochs: list[dict[str, Any]] = []

    for step, item in enumerate(sorted_items):
        dt_utc = _item_dt_utc(item)
        date_str = dt_utc.strftime("%Y-%m-%d")

        if progress_callback is not None:
            progress_callback(step, total, date_str)

        ds: xr.Dataset | None = None
        masked_ds: xr.Dataset | None = None
        layer_da: xr.DataArray | None = None
        try:
            ds = load_scene(
                sensor=sensor,
                item=item,
                bbox_4326=bbox_4326,
                bands=req_bands,
                target_epsg=target_epsg,
                resolution=float(resolution),
                aoi_polygon_4326=aoi_polygon_4326,
            )
            masked_ds, clear_pct = apply_cloud_mask(ds, mask_clouds=use_cloud_mask)
            layer_da = _compute_layer(masked_ds, sensor=sensor, metric_name=idx_key)
            stats = compute_index_stats(layer_da, resolution_m=float(resolution))

            valid_pct = float(stats["valid_pct"])
            if valid_pct < float(min_valid_pct):
                logger.info(
                    "Skipped %s (%s): %.1f%% valid pixels (clear: %.1f%%, min: %.0f%%).",
                    date_str,
                    item.id,
                    valid_pct,
                    clear_pct,
                    min_valid_pct,
                )
                skipped_epochs.append(
                    {
                        "date_str": date_str,
                        "scene_id": item.id,
                        "valid_pct": valid_pct,
                        "reason": "low_valid_pixels",
                    }
                )
            else:
                cloud_raw = item.properties.get("eo:cloud_cover")
                rows.append(
                    {
                        "date": pd.Timestamp(dt_utc),
                        "date_str": date_str,
                        "scene_id": item.id,
                        "platform": str(item.properties.get("platform", "")),
                        "mgrs_tile": str(ds.attrs.get("mgrs_tile", "")),
                        "cloud_cover": float(cloud_raw) if cloud_raw is not None else np.nan,
                        "mean": stats["mean"],
                        "median": stats["median"],
                        "std": stats["std"],
                        "p10": stats["p10"],
                        "p90": stats["p90"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "valid_pct": valid_pct,
                        "valid_pixels": stats["valid_pixels"],
                        "valid_area_ha": stats["valid_area_ha"],
                    }
                )
        except Exception as exc:
            logger.warning("Failed to process epoch %s (%s): %s", date_str, item.id, exc)
            skipped_epochs.append(
                {
                    "date_str": date_str,
                    "scene_id": item.id,
                    "valid_pct": 0.0,
                    "reason": str(exc),
                }
            )
        finally:
            del ds, masked_ds, layer_da
            gc.collect()

    if progress_callback is not None:
        progress_callback(total, total, "done")

    if rows:
        df = pd.DataFrame(rows, columns=TIME_SERIES_COLUMNS).sort_values("date").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=TIME_SERIES_COLUMNS)

    df.attrs["sensor"] = sensor
    df.attrs["index_name"] = idx_key
    df.attrs["skipped_epochs"] = skipped_epochs
    df.attrs["total_requested"] = total
    return df


def _grids_match(da_ref: xr.DataArray, da_other: xr.DataArray) -> bool:
    """Return True if two 2D DataArrays share identical shape, CRS, and coordinates."""
    if da_ref.shape != da_other.shape:
        return False
    if da_ref.rio.crs != da_other.rio.crs:
        return False
    if not np.allclose(da_ref.coords["x"].values, da_other.coords["x"].values, atol=1e-6):
        return False
    if not np.allclose(da_ref.coords["y"].values, da_other.coords["y"].values, atol=1e-6):
        return False
    return True


def compute_change_detection(
    item_t1: pystac.Item,
    item_t2: pystac.Item,
    bbox_4326: Sequence[float],
    index_name: str,
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    mask_clouds: bool = True,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute exact 1-to-1 pixel-aligned bi-temporal change: ``Delta = t2 - t1``.

    Works across Sentinel-2 L2A, Landsat C2 L2 (optical indices), and Sentinel-1
    SAR RTC (backscatter difference in dB, e.g. ``Delta VV [dB]``).
    """
    idx_key = index_name.strip().upper()
    use_cloud_mask = mask_clouds and (sensor != SENSOR_S1)
    req_bands = get_required_bands_for_sensor(sensor, idx_key, include_cloud_mask=use_cloud_mask)

    ds_t1 = load_scene(
        sensor=sensor,
        item=item_t1,
        bbox_4326=bbox_4326,
        bands=req_bands,
        target_epsg=target_epsg,
        resolution=float(resolution),
        aoi_polygon_4326=aoi_polygon_4326,
    )
    masked_t1, valid_pct_t1 = apply_cloud_mask(ds_t1, mask_clouds=use_cloud_mask)
    da_t1 = _compute_layer(masked_t1, sensor=sensor, metric_name=idx_key)
    da_t1.attrs["valid_pixel_pct"] = valid_pct_t1
    del ds_t1, masked_t1
    gc.collect()

    ds_t2 = load_scene(
        sensor=sensor,
        item=item_t2,
        bbox_4326=bbox_4326,
        bands=req_bands,
        target_epsg=target_epsg,
        resolution=float(resolution),
        aoi_polygon_4326=aoi_polygon_4326,
    )
    masked_t2, valid_pct_t2 = apply_cloud_mask(ds_t2, mask_clouds=use_cloud_mask)
    da_t2 = _compute_layer(masked_t2, sensor=sensor, metric_name=idx_key)
    da_t2.attrs["valid_pixel_pct"] = valid_pct_t2
    del ds_t2, masked_t2
    gc.collect()

    if not _grids_match(da_t1, da_t2):
        da_t2 = da_t2.rio.reproject_match(da_t1, resampling=Resampling.bilinear, nodata=np.nan)
        da_t2 = da_t2.astype(np.float32, copy=False)
        da_t2 = da_t2.rio.write_nodata(np.nan, inplace=False)

    arr_t1 = da_t1.values
    arr_t2 = da_t2.values
    both_valid = np.isfinite(arr_t1) & np.isfinite(arr_t2)

    diff_vals = np.full(arr_t1.shape, np.nan, dtype=np.float32)
    np.subtract(arr_t2, arr_t1, out=diff_vals, where=both_valid)
    if sensor == SENSOR_S1:
        np.clip(diff_vals, np.float32(-30.0), np.float32(30.0), out=diff_vals)
        units = "dB"
    else:
        np.clip(diff_vals, np.float32(-2.0), np.float32(2.0), out=diff_vals)
        units = "dimensionless"

    date_t1 = str(da_t1.attrs.get("datetime", ""))[:10]
    date_t2 = str(da_t2.attrs.get("datetime", ""))[:10]

    da_diff = xr.DataArray(
        diff_vals,
        dims=da_t1.dims,
        coords=da_t1.coords,
        name=f"delta_{idx_key}",
        attrs={
            "sensor": sensor,
            "index_name": idx_key,
            "long_name": f"Delta {idx_key} ({date_t2} - {date_t1})",
            "date_t1": date_t1,
            "date_t2": date_t2,
            "scene_id_t1": da_t1.attrs.get("scene_id", ""),
            "scene_id_t2": da_t2.attrs.get("scene_id", ""),
            "resolution_m": float(resolution),
            "units": units,
        },
    )
    if da_t1.rio.crs is not None:
        da_diff = da_diff.rio.write_crs(da_t1.rio.crs, inplace=False)
    da_diff = da_diff.rio.write_nodata(np.nan, inplace=False)

    gc.collect()
    return da_t1, da_t2, da_diff


def compute_change_summary(
    da_diff: xr.DataArray,
    resolution_m: float,
    threshold: float = 0.1,
) -> dict[str, float | int]:
    """Summarize positive, negative, and stable change areas in hectares for ``Delta``."""
    vals = da_diff.values
    total_pixels = int(vals.size)
    valid_vals = vals[np.isfinite(vals)]
    valid_pixels = int(valid_vals.size)
    pixel_area_ha = (float(resolution_m) ** 2) / 10_000.0
    valid_area_ha = float(valid_pixels * pixel_area_ha)
    valid_pct = float(100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    if valid_pixels == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "valid_pixels": 0,
            "valid_pct": 0.0,
            "valid_area_ha": 0.0,
            "pos_area_ha": 0.0,
            "neg_area_ha": 0.0,
            "stable_area_ha": 0.0,
            "pos_pct": 0.0,
            "neg_pct": 0.0,
            "stable_pct": 0.0,
        }

    pos_px = int(np.count_nonzero(valid_vals > threshold))
    neg_px = int(np.count_nonzero(valid_vals < -threshold))
    stable_px = valid_pixels - pos_px - neg_px

    return {
        "mean": float(np.mean(valid_vals)),
        "median": float(np.median(valid_vals)),
        "std": float(np.std(valid_vals)),
        "valid_pixels": valid_pixels,
        "valid_pct": round(valid_pct, 2),
        "valid_area_ha": round(valid_area_ha, 3),
        "pos_area_ha": round(pos_px * pixel_area_ha, 3),
        "neg_area_ha": round(neg_px * pixel_area_ha, 3),
        "stable_area_ha": round(stable_px * pixel_area_ha, 3),
        "pos_pct": round(100.0 * pos_px / valid_pixels, 1),
        "neg_pct": round(100.0 * neg_px / valid_pixels, 1),
        "stable_pct": round(100.0 * stable_px / valid_pixels, 1),
    }
