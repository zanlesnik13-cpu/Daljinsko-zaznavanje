"""Memory-safe multi-sensor time-series extraction and pixel-aligned change detection.

Supports Sentinel-2 L2A, Landsat C2 L2 (1984–present), and Sentinel-1 SAR RTC,
streaming up to 4 satellite epochs in parallel via ``ThreadPoolExecutor(max_workers=4)``
with batch ``gc.collect()`` (16 GB RAM Hugging Face Spaces profile) and exact polygon
clipping (``rioxarray.rio.clip``).
"""

from __future__ import annotations

import gc
import io
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np
import pandas as pd
import pystac
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rasterio.enums import Resampling

from analize import (
    GLACIER_CHANGE_RGBA_MAP,
    SAR_METRICS,
    SPECTRAL_INDICES,
    apply_cloud_mask,
    classify_glacier_change,
    compute_index,
    compute_index_stats,
    compute_water_cleaned_ndsi,
    detect_new_proglacial_lake_mask,
    extract_sar_metric,
    get_required_bands_for_sensor,
)
from geodezija_crs import clip_to_aoi_polygon, get_parallel_workers
from map_overlay import raster_to_map_overlay_png, render_diverging_change_overlay
from stac_loader import (
    MODIS_SENSORS,
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_S2,
    StacItemOrGroup,
    assess_scene_analysis_reliability,
    compute_optical_saturation_pct,
    get_default_reliable_t1_index,
    load_scene,
)

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
    "saturated_pct",
    "analysis_reliable",
    "quality_note",
]


def _item_dt_utc(item: pystac.Item) -> datetime:
    dt = item.datetime
    if dt is None:
        raw = (
            item.properties.get("datetime")
            or item.properties.get("start_datetime")
            or item.properties.get("end_datetime")
        )
        if raw is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    if item.datetime is None:
        item.datetime = dt
        item.properties["datetime"] = dt.isoformat().replace("+00:00", "Z")
    return dt


def _normalize_epoch_groups(
    items: Sequence[StacItemOrGroup],
) -> list[list[pystac.Item]]:
    """Normalize a flat list of STAC items or pre-grouped same-day item lists into chronological date groups."""
    buckets: dict[str, list[pystac.Item]] = {}
    seen_ids: dict[str, set[str]] = {}

    for entry in items:
        sub_items = [entry] if isinstance(entry, pystac.Item) else list(entry)
        for it in sub_items:
            if not isinstance(it, pystac.Item):
                continue
            date_str = _item_dt_utc(it).strftime("%Y-%m-%d")
            bucket_ids = seen_ids.setdefault(date_str, set())
            if it.id not in bucket_ids:
                bucket_ids.add(it.id)
                buckets.setdefault(date_str, []).append(it)

    groups: list[list[pystac.Item]] = []
    for date_str in sorted(buckets.keys()):
        day_items = buckets[date_str]
        groups.append(day_items)
    return groups


def _group_metadata(day_items: Sequence[pystac.Item], sensor: str) -> tuple[datetime, str, str, str, float]:
    """Extract ``(dt_utc, date_str, scene_id, platform, mean_cloud_cover)`` for a same-day item group."""
    item_list = list(day_items)
    dt_utc = max(_item_dt_utc(it) for it in item_list)
    date_str = dt_utc.strftime("%Y-%m-%d")
    primary_id = item_list[0].id
    scene_id = primary_id if len(item_list) == 1 else f"{primary_id} (+{len(item_list) - 1})"
    platforms = sorted({str(it.properties.get("platform", "")).strip() for it in item_list if it.properties.get("platform")})
    platform = " + ".join(platforms)

    if sensor == SENSOR_S1:
        cloud_cover = float("nan")
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
        cloud_cover = float(np.mean(clouds)) if clouds else float("nan")

    return dt_utc, date_str, scene_id, platform, cloud_cover


def _compute_layer(ds: xr.Dataset, sensor: str, metric_name: str) -> xr.DataArray:
    """Compute either an optical spectral index or a SAR backscatter metric."""
    if sensor == SENSOR_S1:
        return extract_sar_metric(ds, metric_name)
    return compute_index(ds, metric_name)


def _compute_raw_swath_coverage_pct(
    ds: xr.Dataset,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> float:
    """Compute raw non-NoData satellite swath coverage [%] before cloud masking.

    When ``aoi_polygon_4326`` is supplied, evaluates coverage relative to the pixels
    inside the clipped polygon rather than the enclosing rectangular bounding box.
    """
    data_vars = [
        str(v)
        for v in ds.data_vars
        if str(v) not in ("SCL", "QA_PIXEL", "qa_pixel")
    ]
    if not data_vars:
        return 0.0

    ref_da = ds[data_vars[0]]
    finite_mask = np.ones(ref_da.shape, dtype=bool)
    for var in data_vars:
        finite_mask &= np.isfinite(ds[var].values)

    finite_px = int(np.count_nonzero(finite_mask))
    denom_px = int(finite_mask.size)

    if aoi_polygon_4326 is not None and denom_px > 0:
        try:
            ones_da = xr.DataArray(
                np.ones(ref_da.shape, dtype=np.float32),
                dims=ref_da.dims,
                coords=ref_da.coords,
            )
            crs = ds.rio.crs or ref_da.rio.crs
            if crs is not None:
                ones_da = ones_da.rio.write_crs(crs, inplace=False)
            ones_da = ones_da.rio.write_nodata(np.nan, inplace=False)
            clipped_ones = clip_to_aoi_polygon(ones_da, aoi_polygon_4326, drop=False)
            poly_px = int(np.count_nonzero(np.isfinite(clipped_ones.values)))
            if poly_px > 0:
                denom_px = poly_px
        except Exception:
            pass

    if denom_px <= 0:
        return 0.0
    return round(float(np.clip(100.0 * finite_px / denom_px, 0.0, 100.0)), 1)


def _resolve_bucket_candidates(
    primary_group: list[pystac.Item],
    sensor: str,
    candidates_by_date: dict[str, list[list[pystac.Item]]] | None,
) -> list[list[pystac.Item]]:
    """Return up to 4 deduplicated candidate date groups for a temporal bucket."""
    _, prim_date, _, _, _ = _group_metadata(primary_group, sensor=sensor)
    raw_cands: list[list[pystac.Item]] = [primary_group]
    if candidates_by_date and prim_date in candidates_by_date:
        for alt_group in candidates_by_date[prim_date]:
            if isinstance(alt_group, list) and alt_group:
                raw_cands.append(alt_group)

    dedup: list[list[pystac.Item]] = []
    seen_dates: set[str] = set()
    for grp in raw_cands:
        try:
            _, d_str, _, _, _ = _group_metadata(grp, sensor=sensor)
        except Exception:
            continue
        if d_str not in seen_dates:
            seen_dates.add(d_str)
            dedup.append(grp)
        if len(dedup) >= 4:
            break
    return dedup or [primary_group]


def extract_time_series(
    items: Sequence[StacItemOrGroup],
    bbox_4326: Sequence[float],
    index_name: str,
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    mask_clouds: bool = True,
    min_valid_pct: float = 75.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_swath_coverage_pct: float = 85.0,
    candidates_by_date: dict[str, list[list[pystac.Item]]] | None = None,
) -> pd.DataFrame:
    """Extract a multi-temporal index or SAR backscatter time series with smart AOI cloud fallback.

    Streams up to 4 temporal buckets in parallel using ``ThreadPoolExecutor(max_workers=4)``
    (with batch ``gc.collect()`` and chronological re-sorting). For each temporal bucket,
    evaluates candidate #1's actual AOI cloud-free valid pixel % (using Sentinel-2 ``SCL``
    or Landsat ``QA_PIXEL``). If candidate #1 fails ``min_swath_coverage_pct`` (default 85%)
    or ``min_valid_pct`` (default 75%), automatically tries candidates #2–#4 from the same
    time bucket until a genuinely cloud-free observation over the AOI is found.
    """
    idx_key = index_name.strip().upper()
    use_cloud_mask = mask_clouds and (sensor != SENSOR_S1) and (sensor not in MODIS_SENSORS)
    req_bands = get_required_bands_for_sensor(sensor, idx_key, include_cloud_mask=use_cloud_mask)

    epoch_groups = _normalize_epoch_groups(items)
    total = len(epoch_groups)
    rows: list[dict[str, Any]] = []
    skipped_epochs: list[dict[str, Any]] = []
    replaced_epochs: list[dict[str, Any]] = []

    def _process_epoch(
        step: int,
        primary_items: list[pystac.Item],
    ) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        _, prim_date_str, prim_scene_id, _, _ = _group_metadata(primary_items, sensor=sensor)
        bucket_cands = _resolve_bucket_candidates(primary_items, sensor, candidates_by_date)
        first_fail_reason = "low_valid_pixels"
        first_fail_pct = 0.0

        for cand_idx, day_items in enumerate(bucket_cands):
            dt_utc, date_str, scene_id, platform, cloud_cover = _group_metadata(day_items, sensor=sensor)
            ds: xr.Dataset | None = None
            masked_ds: xr.Dataset | None = None
            layer_da: xr.DataArray | None = None
            try:
                ds = load_scene(
                    sensor=sensor,
                    item=day_items,
                    bbox_4326=bbox_4326,
                    bands=req_bands,
                    target_epsg=target_epsg,
                    resolution=float(resolution),
                    aoi_polygon_4326=aoi_polygon_4326,
                )
                swath_pct = _compute_raw_swath_coverage_pct(ds, aoi_polygon_4326=aoi_polygon_4326)
                if min_swath_coverage_pct > 0.0 and swath_pct < float(min_swath_coverage_pct):
                    if cand_idx == 0:
                        first_fail_pct = swath_pct
                        first_fail_reason = f"partial_swath ({swath_pct:.1f}% < {min_swath_coverage_pct:.0f}%)"
                    logger.info(
                        "Bucket %s candidate #%d (%s) rejected: partial swath %.1f%% < %.0f%%.",
                        prim_date_str,
                        cand_idx + 1,
                        date_str,
                        swath_pct,
                        min_swath_coverage_pct,
                    )
                    continue

                masked_ds, _clear_pct = apply_cloud_mask(ds, mask_clouds=use_cloud_mask, sensor=sensor)
                layer_da = _compute_layer(masked_ds, sensor=sensor, metric_name=idx_key)
                stats = compute_index_stats(layer_da, resolution_m=float(resolution))

                valid_pct = float(stats["valid_pct"])
                if valid_pct < float(min_valid_pct):
                    if cand_idx == 0:
                        first_fail_pct = valid_pct
                        first_fail_reason = f"aoi_clouds ({valid_pct:.1f}% < {min_valid_pct:.0f}%)"
                    logger.info(
                        "Bucket %s candidate #%d (%s) rejected: %.1f%% AOI clear < %.0f%%.",
                        prim_date_str,
                        cand_idx + 1,
                        date_str,
                        valid_pct,
                        min_valid_pct,
                    )
                    continue

                sat_pct = compute_optical_saturation_pct(masked_ds, sensor=sensor)
                sensor_fam_ts = (
                    "Landsat (30 m)"
                    if sensor == SENSOR_LANDSAT
                    else ("Sentinel-1 SAR" if sensor == SENSOR_S1 else "Sentinel-2 (10–20 m)")
                )
                rel_ok, q_note = assess_scene_analysis_reliability(
                    date_str,
                    sensor_fam_ts,
                    valid_pct,
                    saturated_pct=sat_pct,
                )
                row = {
                    "date": pd.Timestamp(dt_utc),
                    "date_str": date_str,
                    "scene_id": str(ds.attrs.get("scene_id", scene_id)),
                    "platform": str(ds.attrs.get("platform", platform)),
                    "mgrs_tile": str(ds.attrs.get("mgrs_tile", "")),
                    "cloud_cover": cloud_cover,
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
                    "saturated_pct": sat_pct,
                    "analysis_reliable": rel_ok,
                    "quality_note": q_note,
                }
                replaced_entry = None
                if cand_idx > 0:
                    replaced_entry = {
                        "orig_date": prim_date_str,
                        "used_date": date_str,
                        "orig_valid_pct": round(first_fail_pct, 1),
                        "used_valid_pct": round(valid_pct, 1),
                        "candidate_rank": cand_idx + 1,
                    }
                    logger.info(
                        "Replaced cloudy epoch %s (%.1f%%) with clear candidate #%d: %s (%.1f%% clear).",
                        prim_date_str,
                        first_fail_pct,
                        cand_idx + 1,
                        date_str,
                        valid_pct,
                    )
                return step, date_str, row, replaced_entry, None
            except Exception as exc:
                if cand_idx == 0:
                    first_fail_reason = str(exc)
                logger.warning("Failed candidate #%d %s (%s): %s", cand_idx + 1, date_str, scene_id, exc)
            finally:
                del ds, masked_ds, layer_da

        skipped_entry = {
            "date_str": prim_date_str,
            "scene_id": prim_scene_id,
            "valid_pct": round(first_fail_pct, 1),
            "reason": first_fail_reason,
        }
        return step, prim_date_str, None, None, skipped_entry

    if total > 0:
        batch_size = get_parallel_workers()
        completed = 0
        with ThreadPoolExecutor(max_workers=min(batch_size, total)) as executor:
            for batch_start in range(0, total, batch_size):
                batch = epoch_groups[batch_start : batch_start + batch_size]
                future_map = {
                    executor.submit(_process_epoch, batch_start + offset, grp): (batch_start + offset, grp)
                    for offset, grp in enumerate(batch)
                }
                for fut in as_completed(future_map):
                    step_idx, orig_grp = future_map[fut]
                    completed += 1
                    try:
                        _step, done_date, row, rep_entry, skip_entry = fut.result()
                        if row is not None:
                            rows.append(row)
                        if rep_entry is not None:
                            replaced_epochs.append(rep_entry)
                        if skip_entry is not None:
                            skipped_epochs.append(skip_entry)
                        if progress_callback is not None:
                            progress_callback(min(completed, total), total, done_date)
                    except Exception as exc:
                        _, err_date, err_id, _, _ = _group_metadata(orig_grp, sensor=sensor)
                        logger.warning("Time-series worker failed for %s (%s): %s", err_date, err_id, exc)
                        skipped_epochs.append(
                            {
                                "date_str": err_date,
                                "scene_id": err_id,
                                "valid_pct": 0.0,
                                "reason": str(exc),
                            }
                        )
                        if progress_callback is not None:
                            progress_callback(min(completed, total), total, err_date)
                gc.collect()

    if progress_callback is not None:
        progress_callback(total, total, "done")

    skipped_epochs.sort(key=lambda d: str(d.get("date_str", "")))
    replaced_epochs.sort(key=lambda d: str(d.get("orig_date", "")))

    if rows:
        df = pd.DataFrame(rows, columns=TIME_SERIES_COLUMNS).sort_values("date").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=TIME_SERIES_COLUMNS)

    df.attrs["sensor"] = sensor
    df.attrs["index_name"] = idx_key
    df.attrs["skipped_epochs"] = skipped_epochs
    df.attrs["replaced_epochs"] = replaced_epochs
    df.attrs["total_requested"] = total
    gc.collect()
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
    item_t1: StacItemOrGroup,
    item_t2: StacItemOrGroup,
    bbox_4326: Sequence[float],
    index_name: str,
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    mask_clouds: bool = True,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute exact 1-to-1 pixel-aligned bi-temporal change: ``Delta = t2 - t1``.

    Automatically mosaics all same-day tiles when ``item_t1`` / ``item_t2`` are
    lists of STAC items. Works across Sentinel-2 L2A, Landsat C2 L2, and
    Sentinel-1 SAR RTC (backscatter difference in dB, e.g. ``Delta VV [dB]``).
    """
    idx_key = index_name.strip().upper()
    use_cloud_mask = mask_clouds and (sensor != SENSOR_S1) and (sensor not in MODIS_SENSORS)
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
    elif idx_key == "LST":
        np.clip(diff_vals, np.float32(-60.0), np.float32(60.0), out=diff_vals)
        units = "°C"
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


def align_and_compute_delta(
    da_t1: xr.DataArray,
    da_t2: xr.DataArray,
    *,
    target_epsg: int | str = "EPSG:3794",
    index_name: str = "NDVI",
    threshold: float = 0.15,
    max_width: int = 1200,
) -> dict[str, Any]:
    """Align two metric-CRS DataArrays of potentially different resolutions (e.g. 30 m Landsat & 10 m S2)
    and compute pixel-wise ``Δ = t2 - t1``, area statistics, and a diverging RGBA overlay.
    """
    ref_da = da_t1
    other_da = da_t2

    # Ensure both arrays carry CRS metadata for rioxarray reproject_match
    if ref_da.rio.crs is None:
        ref_da = ref_da.rio.write_crs(target_epsg, inplace=False)
    if other_da.rio.crs is None:
        other_da = other_da.rio.write_crs(target_epsg, inplace=False)

    if not _grids_match(ref_da, other_da):
        other_aligned = other_da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
    else:
        other_aligned = other_da

    diff_vals = (
        np.asarray(other_aligned.values, dtype=np.float32)
        - np.asarray(ref_da.values, dtype=np.float32)
    )
    diff_da = xr.DataArray(
        diff_vals,
        dims=ref_da.dims,
        coords=ref_da.coords,
        name=f"Delta_{str(index_name or 'IDX').upper()}",
    )
    diff_da = diff_da.rio.write_crs(ref_da.rio.crs, inplace=False)
    diff_da = diff_da.rio.write_nodata(np.nan, inplace=False)

    # Infer exact metric pixel resolution from reference x coordinates
    x_vals = np.asarray(ref_da.coords["x"].values, dtype=np.float64)
    if x_vals.size >= 2:
        aligned_res_m = float(abs(x_vals[1] - x_vals[0]))
    else:
        aligned_res_m = 30.0

    ch_summary = compute_change_summary(
        diff_da,
        resolution_m=aligned_res_m,
        threshold=float(threshold),
    )
    delta_uri, delta_bnds, delta_vmin, delta_vmax, delta_cmap = render_diverging_change_overlay(
        diff_da,
        src_epsg=target_epsg,
        index_name=str(index_name or "NDVI").upper(),
        max_width=max_width,
    )
    return {
        "diff_da": diff_da,
        "change_summary": ch_summary,
        "delta_overlay_uri": delta_uri,
        "delta_overlay_bounds": delta_bnds,
        "delta_vmin": delta_vmin,
        "delta_vmax": delta_vmax,
        "delta_cmap": delta_cmap,
        "aligned_resolution_m": round(aligned_res_m, 2),
    }


def _downsample_da_to_small_grid(
    da: xr.DataArray,
    *,
    max_dim: int = 280,
    fallback_crs: int | str = "EPSG:3794",
) -> dict[str, Any]:
    """Downsample a 2D metric DataArray to <= ``max_dim × max_dim`` (~150 KB) for instant on-the-fly Δ."""
    h, w = int(da.shape[-2]), int(da.shape[-1])
    step_y = max(1, int(np.ceil(h / float(max_dim))))
    step_x = max(1, int(np.ceil(w / float(max_dim))))
    sub = da.isel(y=slice(0, None, step_y), x=slice(0, None, step_x))
    crs_val = str(da.rio.crs) if da.rio.crs is not None else str(fallback_crs)
    return {
        "values": np.asarray(sub.values, dtype=np.float32),
        "x": np.asarray(sub.coords["x"].values, dtype=np.float64),
        "y": np.asarray(sub.coords["y"].values, dtype=np.float64),
        "crs": crs_val,
    }


def _small_grid_to_da(grid_dict: dict[str, Any], fallback_crs: int | str = "EPSG:3794") -> xr.DataArray:
    """Reconstruct a georeferenced float32 DataArray from a lightweight cached grid dict."""
    arr = np.asarray(grid_dict["values"], dtype=np.float32)
    x_c = np.asarray(grid_dict["x"], dtype=np.float64)
    y_c = np.asarray(grid_dict["y"], dtype=np.float64)
    crs_str = str(grid_dict.get("crs") or fallback_crs)
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": y_c, "x": x_c})
    da = da.rio.write_crs(crs_str, inplace=False)
    da = da.rio.write_nodata(np.nan, inplace=False)
    return da


def compute_dynamic_frame_delta(
    frame_t1: dict[str, Any],
    frame_t2: dict[str, Any],
    *,
    target_epsg: int | str = "EPSG:3794",
    index_name: str = "NDSI",
    threshold: float | None = None,
    max_width: int = 1200,
) -> dict[str, Any] | None:
    """Recompute aligned ``Δ (t2 - t1)``, area stats, and (for NDSI) water-aware glacier change on-the-fly.

    Works instantaneously from lightweight cached grids in ``frames`` with ZERO extra STAC calls
    and supports mixed-resolution pairs (e.g. 30 m Landsat t1 + 10 m Sentinel-2 t2).
    """
    g1 = frame_t1.get("index_grid_small")
    g2 = frame_t2.get("index_grid_small")
    if not (isinstance(g1, dict) and isinstance(g2, dict)):
        return None

    idx_u = str(index_name or "NDVI").strip().upper()
    da1 = _small_grid_to_da(g1, fallback_crs=target_epsg)
    da2 = _small_grid_to_da(g2, fallback_crs=target_epsg)
    if not _grids_match(da1, da2):
        da2_al = da2.rio.reproject_match(da1, resampling=Resampling.bilinear)
    else:
        da2_al = da2

    x_vals = np.asarray(da1.coords["x"].values, dtype=np.float64)
    aligned_res_m = float(abs(x_vals[1] - x_vals[0])) if x_vals.size >= 2 else 30.0
    eff_thr = float(threshold if threshold is not None else (2.0 if idx_u in ("LST", "VV", "VH") else 0.12))

    glacier_class_uri: str | None = None
    glacier_class_bnds: list[list[float]] | None = None
    glacier_change_stats: dict[str, float] | None = None

    if idx_u == "NDSI":
        # Water-aware GLIMS cleaning: proglacial lakes (high NDWI / low NIR) become -0.5 instead of +0.6!
        w1_da = _small_grid_to_da(frame_t1["ndwi_grid_small"], fallback_crs=target_epsg) if isinstance(frame_t1.get("ndwi_grid_small"), dict) else None
        w2_da = _small_grid_to_da(frame_t2["ndwi_grid_small"], fallback_crs=target_epsg) if isinstance(frame_t2.get("ndwi_grid_small"), dict) else None
        r1_da = _small_grid_to_da(frame_t1["nir_grid_small"], fallback_crs=target_epsg) if isinstance(frame_t1.get("nir_grid_small"), dict) else None
        r2_da = _small_grid_to_da(frame_t2["nir_grid_small"], fallback_crs=target_epsg) if isinstance(frame_t2.get("nir_grid_small"), dict) else None

        if w1_da is not None and not _grids_match(da1, w1_da):
            w1_da = w1_da.rio.reproject_match(da1, resampling=Resampling.bilinear)
        if w2_da is not None and not _grids_match(da1, w2_da):
            w2_da = w2_da.rio.reproject_match(da1, resampling=Resampling.bilinear)
        if r1_da is not None and not _grids_match(da1, r1_da):
            r1_da = r1_da.rio.reproject_match(da1, resampling=Resampling.bilinear)
        if r2_da is not None and not _grids_match(da1, r2_da):
            r2_da = r2_da.rio.reproject_match(da1, resampling=Resampling.bilinear)

        clean_1 = compute_water_cleaned_ndsi(da1, ndwi=w1_da, nir=r1_da)
        clean_2 = compute_water_cleaned_ndsi(da2_al, ndwi=w2_da, nir=r2_da)
        diff_vals = (clean_2 - clean_1).astype(np.float32)

        # Direct ΔNDWI proglacial lake gain detection (catches turbid glacial-milk lakes & debris-ice -> lake)
        new_lake_mask = detect_new_proglacial_lake_mask(w1_da, w2_da)
        if new_lake_mask is not None:
            diff_vals = np.where(new_lake_mask, np.float32(-0.60), diff_vals).astype(np.float32)

        cls_u8, gl_stats = classify_glacier_change(
            da1,
            da2_al,
            ndwi_t1=w1_da,
            ndwi_t2=w2_da,
            nir_t1=r1_da,
            nir_t2=r2_da,
            resolution_m=aligned_res_m,
        )
        cls_da = xr.DataArray(cls_u8, dims=da1.dims, coords=da1.coords, name="Glacier_Change_Class")
        cls_da = cls_da.rio.write_crs(da1.rio.crs, inplace=False).rio.write_nodata(0, inplace=False)
        glacier_class_uri, glacier_class_bnds = raster_to_map_overlay_png(
            cls_da,
            src_epsg=target_epsg,
            is_rgb=False,
            discrete_rgba_map=GLACIER_CHANGE_RGBA_MAP,
            max_width=max_width,
        )
        glacier_change_stats = gl_stats
    else:
        diff_vals = (
            np.asarray(da2_al.values, dtype=np.float32)
            - np.asarray(da1.values, dtype=np.float32)
        )

    diff_da = xr.DataArray(diff_vals, dims=da1.dims, coords=da1.coords, name=f"Delta_{idx_u}")
    diff_da = diff_da.rio.write_crs(da1.rio.crs, inplace=False).rio.write_nodata(np.nan, inplace=False)
    ch_summary = compute_change_summary(diff_da, resolution_m=aligned_res_m, threshold=eff_thr)
    delta_uri, delta_bnds, delta_vmin, delta_vmax, delta_cmap = render_diverging_change_overlay(
        diff_da,
        src_epsg=target_epsg,
        index_name=idx_u,
        max_width=max_width,
    )
    return {
        "diff_da": diff_da,
        "change_summary": ch_summary,
        "delta_overlay_uri": delta_uri,
        "delta_overlay_bounds": delta_bnds,
        "delta_vmin": delta_vmin,
        "delta_vmax": delta_vmax,
        "delta_cmap": delta_cmap,
        "glacier_class_uri": glacier_class_uri,
        "glacier_class_bounds": glacier_class_bnds,
        "glacier_change_stats": glacier_change_stats,
        "aligned_resolution_m": round(aligned_res_m, 2),
    }


# ---------------------------------------------------------------------------
# Visual Timelapse (PIL + Animated GIF) & Before/After Swipe Frame Rendering
# ---------------------------------------------------------------------------
def _load_stamp_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a crisp sans-serif TrueType font bundled with Matplotlib or system fonts."""
    candidates = [
        os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _format_platform_stamp(platform: str, sensor: str) -> str:
    """Format a compact, readable sensor/platform label for timelapse banners."""
    raw = str(platform or "").strip()
    if raw:
        low = raw.lower()
        if "landsat-" in low:
            return raw.replace("-", " ").title()
        if "sentinel-2" in low:
            return raw.replace("sentinel-", "Sentinel-").upper().replace("SENTINEL-", "Sentinel-")
        if "sentinel-1" in low:
            return raw.replace("sentinel-", "Sentinel-").upper().replace("SENTINEL-", "Sentinel-")
        return raw
    if sensor == SENSOR_LANDSAT:
        return "Landsat C2 L2"
    if sensor == SENSOR_S1:
        return "Sentinel-1 SAR"
    return "Sentinel-2 L2A"


def _ensure_north_up(arr: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Ensure row 0 corresponds to North (max Y) so PIL images match map orientation."""
    if y_coords.size > 1 and float(y_coords[0]) < float(y_coords[-1]):
        return arr[::-1, ...]
    return arr


def _resize_pil_frame(
    img: Image.Image,
    max_width: int = 700,
    min_width: int = 480,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Resize PIL image preserving aspect ratio (or exact ``target_size`` for GIF alignment)."""
    if target_size is not None:
        if img.size == target_size:
            return img
        resamp = (
            Image.Resampling.LANCZOS
            if target_size[0] < img.width
            else Image.Resampling.NEAREST
        )
        return img.resize(target_size, resample=resamp)

    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    if w > max_width:
        scale = float(max_width) / float(w)
        new_size = (max_width, max(1, int(round(h * scale))))
        return img.resize(new_size, resample=Image.Resampling.LANCZOS)

    if w < min_width:
        scale = float(min_width) / float(w)
        new_h = max(1, int(round(h * scale)))
        if new_h > max_width:
            scale = float(max_width) / float(h)
            new_size = (max(1, int(round(w * scale))), max_width)
        else:
            new_size = (min_width, new_h)
        return img.resize(new_size, resample=Image.Resampling.NEAREST)

    return img


TIMELAPSE_CANVAS_W: int = 640
TIMELAPSE_CANVAS_H: int = 520
TIMELAPSE_BANNER_H: int = 44


def _stamp_banner_on_pil(
    img: Image.Image,
    stamp_text: str,
    *,
    canvas_w: int = TIMELAPSE_CANVAS_W,
    canvas_h: int = TIMELAPSE_CANVAS_H,
    banner_h: int = TIMELAPSE_BANNER_H,
) -> Image.Image:
    """Place the full-extent scene proportionally on a fixed 640×520 canvas with a 44 px top header bar.

    Never crops or stretches mixed-resolution (e.g. Landsat 30 m + Sentinel-2 10 m) frames, and keeps
    the date/platform banner 100% visible above the satellite image.
    """
    body_h = max(64, int(canvas_h) - int(banner_h))
    canvas = Image.new("RGB", (int(canvas_w), int(canvas_h)), (226, 232, 240))

    # Proportionally fit the ENTIRE scene into (canvas_w, body_h) without cropping
    fitted = ImageOps.contain(img.convert("RGB"), (int(canvas_w), body_h), method=Image.Resampling.LANCZOS)
    off_x = (int(canvas_w) - fitted.width) // 2
    off_y = int(banner_h) + (body_h - fitted.height) // 2
    canvas.paste(fitted, (off_x, off_y))

    # Crisp 44 px dark slate (#0F172A) top header bar with cartographic teal accent
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, int(canvas_w), int(banner_h)], fill=(15, 23, 42))
    draw.rectangle([0, int(banner_h) - 3, int(canvas_w), int(banner_h)], fill=(15, 118, 110))

    if stamp_text:
        font = _load_stamp_font(size=16)
        bbox = draw.textbbox((0, 0), stamp_text, font=font)
        text_h = bbox[3] - bbox[1]
        ty = max(4, (int(banner_h) - 3 - text_h) // 2 - bbox[1])
        draw.text((14, ty), stamp_text, fill=(248, 250, 252), font=font)

    return canvas


def _stretch_rgb_for_pil(
    ds: xr.Dataset,
    bands: Sequence[str],
    y_coords: np.ndarray,
) -> np.ndarray:
    """Compose a north-up uint8 (H, W, 3) array using clean 2–98% stretch on finite reflectance in (0, 1]."""
    channels: list[np.ndarray] = []
    valid = np.ones(ds[bands[0]].shape, dtype=bool)
    for b in bands:
        arr = ds[b].values.astype(np.float32, copy=False)
        valid &= np.isfinite(arr)
        channels.append(arr)

    bg_rgb = (226, 232, 240)  # Neutral cartographic slate #E2E8F0
    out_ch: list[np.ndarray] = []
    for idx, arr in enumerate(channels):
        arr_c = np.where(np.isfinite(arr), np.clip(arr, np.float32(0.0), np.float32(1.0)), np.nan)
        fin = arr_c[np.isfinite(arr_c) & (arr_c > 0.0) & (arr_c <= 1.0)]
        if fin.size == 0:
            fin = arr_c[np.isfinite(arr_c)]
        if fin.size == 0:
            stretched = np.zeros_like(arr, dtype=np.float32)
        else:
            lo, hi = np.percentile(fin, [2.0, 98.0])
            if hi <= lo:
                lo, hi = float(fin.min()), float(fin.max())
            denom = max(float(hi - lo), 1e-6)
            stretched = np.clip((arr_c - np.float32(lo)) / np.float32(denom), 0.0, 1.0)
        u8 = np.where(valid, np.clip(np.round(stretched * 255.0), 0, 255), bg_rgb[idx]).astype(
            np.uint8
        )
        out_ch.append(u8)

    rgb = np.stack(out_ch, axis=-1)
    return _ensure_north_up(rgb, y_coords)


def render_dataarray_to_pil(
    da: xr.DataArray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    stamp_text: str | None = None,
    max_width: int = 700,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Render a 2D georeferenced DataArray into a crisp north-up PIL RGB Image."""
    vals = da.values.astype(np.float32, copy=False)
    y_coords = np.asarray(da.coords["y"].values, dtype=np.float64)
    vals = _ensure_north_up(vals, y_coords)

    finite = np.isfinite(vals)
    denom = max(float(vmax - vmin), 1e-6)
    norm = np.clip((np.nan_to_num(vals, nan=vmin) - np.float32(vmin)) / np.float32(denom), 0.0, 1.0)

    cmap = matplotlib.colormaps[cmap_name]
    rgba_f = cmap(norm)  # (H, W, 4) in [0, 1]
    rgb_u8 = np.clip(np.round(rgba_f[..., :3] * 255.0), 0, 255).astype(np.uint8)
    # Neutral light slate #E2E8F0 for NaN / clipped pixels
    rgb_u8[~finite] = np.array([226, 232, 240], dtype=np.uint8)

    img = Image.fromarray(rgb_u8, mode="RGB")
    if stamp_text:
        img = _stamp_banner_on_pil(img, stamp_text)
    else:
        img = _resize_pil_frame(img, max_width=max_width, target_size=target_size)
    return img


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL Image to compact PNG bytes in memory."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def compile_timelapse_gif(
    frames_png_bytes: Sequence[bytes],
    duration_ms: int = 800,
    max_canvas: tuple[int, int] = (TIMELAPSE_CANVAS_W, TIMELAPSE_CANVAS_H),
) -> bytes:
    """Compile a sequence of PNG frame byte buffers into a uniform, uncropped looping animated GIF."""
    if not frames_png_bytes:
        raise ValueError("Cannot compile timelapse GIF from an empty frame list.")

    raw_imgs: list[Image.Image] = []
    max_w, max_h = int(max_canvas[0]), int(max_canvas[1])
    for raw in frames_png_bytes:
        with Image.open(io.BytesIO(raw)) as im:
            raw_imgs.append(im.convert("RGB"))

    # Choose uniform target size <= max_canvas preserving full extent without cropping
    first_w, first_h = raw_imgs[0].size
    if first_w <= max_w and first_h <= max_h and all(im.size == (first_w, first_h) for im in raw_imgs):
        target_sz = (first_w, first_h)
    else:
        target_sz = (max_w, max_h)

    pil_frames: list[Image.Image] = []
    for frame_rgb in raw_imgs:
        if frame_rgb.size != target_sz:
            frame_rgb = ImageOps.pad(
                frame_rgb,
                target_sz,
                method=Image.Resampling.LANCZOS,
                color=(226, 232, 240),
            )
        pil_frames.append(frame_rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(max(100, duration_ms)),
        loop=0,
        optimize=True,
    )
    gif_bytes = buf.getvalue()
    del raw_imgs, pil_frames
    gc.collect()
    return gif_bytes


def compile_timelapse_gif_fps(
    frames_png_bytes: Sequence[bytes],
    fps: float = 1.5,
) -> bytes:
    """Compile cached PNG frame bytes into a looping animated GIF at ``fps`` frames/sec."""
    safe_fps = max(0.25, min(float(fps), 10.0))
    duration_ms = int(round(1000.0 / safe_fps))
    return compile_timelapse_gif(frames_png_bytes, duration_ms=duration_ms)


def classify_frame_sensor_group(frame: dict[str, Any]) -> str:
    """Classify a timelapse frame dict into a concise sensor family group."""
    raw = " ".join(
        str(frame.get(k) or "")
        for k in ("sensor_family", "platform", "sensor")
    ).lower()
    if "modis" in raw or "mod09" in raw or "myd09" in raw or "mod11" in raw:
        return "MODIS"
    if "landsat" in raw or "lt05" in raw or "le07" in raw or "lc08" in raw or "lc09" in raw:
        return "Landsat"
    if "sentinel-1" in raw or "sar" in raw or "s1" in raw:
        return "Sentinel-1"
    if "sentinel-2" in raw or "s2" in raw:
        return "Sentinel-2"
    return "Other"


def _short_sensor_tag(frame: dict[str, Any]) -> str:
    """Return a compact 2–5 char sensor tag for touch-friendly timelapse pills (e.g. L5/7, MODIS, S2)."""
    plat = str(frame.get("platform") or frame.get("sensor_family") or frame.get("sensor") or "").lower()
    if "modis" in plat:
        return "MODIS"
    if "landsat-5/7" in plat:
        return "L5/7"
    if "landsat-8/9" in plat:
        return "L8/9"
    if "landsat-5" in plat or "lt05" in plat:
        return "L5"
    if "landsat-7" in plat or "le07" in plat:
        return "L7"
    if "landsat-8" in plat or "lc08" in plat:
        return "L8"
    if "landsat-9" in plat or "lc09" in plat:
        return "L9"
    if "landsat" in plat:
        return "Landsat"
    if "sentinel-1" in plat or "sar" in plat:
        return "S1"
    if "sentinel-2" in plat or "s2" in plat:
        return "S2"
    return "EO"


def format_timelapse_frame_pill_label(frame: dict[str, Any], idx: int = 0) -> str:
    """Format a compact, unique pill label ``YYYY-MM-DD · <sensor>`` for a timelapse frame."""
    d_raw = str(frame.get("date") or frame.get("date_str") or f"#{idx + 1}").strip()[:10]
    tag = _short_sensor_tag(frame)
    arch = " ⚠️" if frame.get("analysis_reliable") is False else ""
    return f"{d_raw} · {tag}{arch}"


def filter_timelapse_frames(
    frames: Sequence[dict[str, Any]],
    *,
    included_indices: Sequence[int] | None = None,
    reliable_only: bool = False,
    exclude_sensor_groups: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter timelapse frames while strictly preserving chronological order."""
    excl = {str(g).strip().lower() for g in (exclude_sensor_groups or []) if str(g).strip()}
    inc_set = set(int(i) for i in included_indices) if included_indices is not None else None
    out: list[dict[str, Any]] = []
    for i, fr in enumerate(frames):
        if inc_set is not None and i not in inc_set:
            continue
        if reliable_only and fr.get("analysis_reliable") is False:
            continue
        if excl and classify_frame_sensor_group(fr).lower() in excl:
            continue
        out.append(fr)
    out.sort(key=lambda f: str(f.get("date") or f.get("date_str") or ""))
    return out


def rebuild_timelapse_gif_from_frames(
    frames: Sequence[dict[str, Any]],
    *,
    mode: str = "rgb",
    fps: float = 1.5,
    duration_ms: int | None = None,
) -> tuple[bytes | None, bytes | None]:
    """Rebuild an animated GIF (or single preview PNG) instantaneously from cached frame bytes.

    Returns ``(gif_bytes_or_None, single_preview_png_or_None)``.
    - 0 valid frames -> ``(None, None)``
    - 1 valid frame  -> ``(None, png_bytes)``
    - >= 2 frames    -> ``(gif_bytes, png_bytes[0])``
    """
    pngs: list[bytes] = []
    m_low = str(mode or "rgb").strip().lower()
    for fr in frames:
        raw_b: bytes | None = None
        if m_low == "rgb":
            raw_b = fr.get("rgb_png_bytes") or fr.get("png_bytes")
        elif m_low == "index":
            raw_b = fr.get("index_png_bytes") or fr.get("png_bytes")
        else:
            raw_b = (fr.get("extra_pngs") or {}).get(mode) or fr.get("png_bytes") or fr.get("index_png_bytes")
        if isinstance(raw_b, (bytes, bytearray)) and len(raw_b) > 0:
            pngs.append(bytes(raw_b))

    if not pngs:
        return None, None
    if len(pngs) == 1:
        return None, pngs[0]
    if duration_ms is not None and int(duration_ms) > 0:
        gif_b = compile_timelapse_gif(pngs, duration_ms=int(duration_ms))
    else:
        gif_b = compile_timelapse_gif_fps(pngs, fps=float(fps))
    return gif_b, pngs[0]


def render_change_swipe_pair(
    da_t1: xr.DataArray,
    da_t2: xr.DataArray,
    index_key: str,
    is_sar: bool = False,
    max_width: int = 720,
) -> tuple[bytes, bytes]:
    """Render pixel-aligned ``t1`` and ``t2`` DataArrays to identical-dimension PNG bytes for swipe comparison."""
    spec = SAR_METRICS[index_key] if is_sar else SPECTRAL_INDICES[index_key]
    vmin, vmax = float(spec["vmin"]), float(spec["vmax"])
    cmap_name = str(spec["cmap"])

    img_t1 = render_dataarray_to_pil(
        da_t1, cmap_name=cmap_name, vmin=vmin, vmax=vmax, stamp_text=None, max_width=max_width
    )
    img_t2 = render_dataarray_to_pil(
        da_t2,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        stamp_text=None,
        max_width=max_width,
        target_size=img_t1.size,
    )
    return pil_to_png_bytes(img_t1), pil_to_png_bytes(img_t2)


def generate_timelapse_frames(
    items: Sequence[StacItemOrGroup],
    bbox_4326: Sequence[float],
    layer_mode: str = "RGB",
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
    mask_clouds: bool = True,
    min_valid_pct: float = 75.0,
    duration_ms: int = 800,
    max_width: int = 700,
    progress_callback: Callable[[int, int, str], None] | None = None,
    min_swath_coverage_pct: float = 85.0,
    candidates_by_date: dict[str, list[list[pystac.Item]]] | None = None,
) -> dict[str, Any]:
    """Generate a visual timelapse with 4-thread parallel streaming and 4-candidate AOI cloud fallback.

    Streams up to 4 satellite dates in parallel via ``ThreadPoolExecutor(max_workers=4)``
    with batch ``gc.collect()`` and chronological re-sorting before GIF compilation.
    Evaluates actual cloud-free pixel % inside the AOI (even for RGB / NIR frames via
    Sentinel-2 ``SCL`` or Landsat ``QA_PIXEL``). If candidate #1 has ``< min_valid_pct``
    (default 75%) cloud-free pixels over the AOI, automatically tries candidates #2–#4
    from the same time bucket so cloudy frames and empty gaps are eliminated.
    """
    mode = layer_mode.strip().upper()
    is_sar = sensor == SENSOR_S1
    is_rgb_composite = mode in ("RGB", "NIR") and not is_sar
    mask_band = "QA_PIXEL" if sensor == SENSOR_LANDSAT else "SCL"

    if is_sar:
        if mode not in SAR_METRICS:
            mode = "VV"
        req_bands = list(SAR_METRICS[mode]["pols"])
        render_bands = list(req_bands)
        use_cloud_mask = False
    elif mode == "RGB":
        render_bands = ["B04", "B03", "B02"]
        use_cloud_mask = bool(mask_clouds)
        req_bands = render_bands + ([mask_band] if use_cloud_mask else [])
    elif mode == "NIR":
        render_bands = ["B08", "B04", "B03"]
        use_cloud_mask = bool(mask_clouds)
        req_bands = render_bands + ([mask_band] if use_cloud_mask else [])
    else:
        use_cloud_mask = bool(mask_clouds)
        req_bands = get_required_bands_for_sensor(sensor, mode, include_cloud_mask=use_cloud_mask)
        render_bands = list(req_bands)

    epoch_groups = _normalize_epoch_groups(items)
    total = len(epoch_groups)

    frames: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    replaced: list[dict[str, Any]] = []

    def _process_tl_epoch(
        step: int,
        primary_items: list[pystac.Item],
    ) -> tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        _, prim_date_str, prim_scene_id, _, _ = _group_metadata(primary_items, sensor=sensor)
        bucket_cands = _resolve_bucket_candidates(primary_items, sensor, candidates_by_date)
        first_fail_reason = "low_valid_pixels"
        first_fail_pct = 0.0

        for cand_idx, day_items in enumerate(bucket_cands):
            _, date_str, scene_id, platform_raw, _ = _group_metadata(day_items, sensor=sensor)
            ds: xr.Dataset | None = None
            masked_ds: xr.Dataset | None = None
            layer_da: xr.DataArray | None = None
            img: Image.Image | None = None

            try:
                ds = load_scene(
                    sensor=sensor,
                    item=day_items,
                    bbox_4326=bbox_4326,
                    bands=req_bands,
                    target_epsg=target_epsg,
                    resolution=float(resolution),
                    aoi_polygon_4326=aoi_polygon_4326,
                )

                platform_lbl = _format_platform_stamp(str(ds.attrs.get("platform", platform_raw)), sensor)
                stamp = f"{date_str} | {platform_lbl} | {mode}"
                merged_scene_id = str(ds.attrs.get("scene_id", scene_id))

                swath_pct = _compute_raw_swath_coverage_pct(ds, aoi_polygon_4326=aoi_polygon_4326)
                if min_swath_coverage_pct > 0.0 and swath_pct < float(min_swath_coverage_pct):
                    if cand_idx == 0:
                        first_fail_pct = swath_pct
                        first_fail_reason = f"partial_swath ({swath_pct:.1f}% < {min_swath_coverage_pct:.0f}%)"
                    continue

                if is_rgb_composite:
                    if use_cloud_mask:
                        _, clear_pct = apply_cloud_mask(ds, mask_clouds=True, sensor=sensor)
                        valid_pct = float(clear_pct)
                    else:
                        valid = np.ones(ds[render_bands[0]].shape, dtype=bool)
                        for b in render_bands:
                            valid &= np.isfinite(ds[b].values)
                        valid_pct = float(100.0 * np.count_nonzero(valid) / max(valid.size, 1))

                    if valid_pct < float(min_valid_pct):
                        if cand_idx == 0:
                            first_fail_pct = valid_pct
                            first_fail_reason = f"aoi_clouds ({valid_pct:.1f}% < {min_valid_pct:.0f}%)"
                        continue

                    y_coords = np.asarray(ds.coords["y"].values, dtype=np.float64)
                    rgb_u8 = _stretch_rgb_for_pil(ds, render_bands, y_coords)
                    img = Image.fromarray(rgb_u8, mode="RGB")
                    img = _resize_pil_frame(img, max_width=max_width, target_size=None)
                    img = _stamp_banner_on_pil(img, stamp)
                else:
                    masked_ds, _ = apply_cloud_mask(ds, mask_clouds=use_cloud_mask, sensor=sensor)
                    layer_da = _compute_layer(masked_ds, sensor=sensor, metric_name=mode)
                    stats = compute_index_stats(layer_da, resolution_m=float(resolution))
                    valid_pct = float(stats["valid_pct"])
                    if valid_pct < float(min_valid_pct):
                        if cand_idx == 0:
                            first_fail_pct = valid_pct
                            first_fail_reason = f"aoi_clouds ({valid_pct:.1f}% < {min_valid_pct:.0f}%)"
                        continue

                    spec = SAR_METRICS[mode] if is_sar else SPECTRAL_INDICES[mode]
                    img = render_dataarray_to_pil(
                        layer_da,
                        cmap_name=str(spec["cmap"]),
                        vmin=float(spec["vmin"]),
                        vmax=float(spec["vmax"]),
                        stamp_text=stamp,
                        max_width=max_width,
                        target_size=None,
                    )

                frame_entry = {
                    "date_str": date_str,
                    "platform": platform_lbl,
                    "scene_id": merged_scene_id,
                    "valid_pct": round(valid_pct, 1),
                    "png_bytes": pil_to_png_bytes(img),
                }
                replaced_entry = None
                if cand_idx > 0:
                    replaced_entry = {
                        "orig_date": prim_date_str,
                        "used_date": date_str,
                        "orig_valid_pct": round(first_fail_pct, 1),
                        "used_valid_pct": round(valid_pct, 1),
                        "candidate_rank": cand_idx + 1,
                    }
                return step, date_str, frame_entry, replaced_entry, None
            except Exception as exc:
                if cand_idx == 0:
                    first_fail_reason = str(exc)
                logger.warning("Timelapse candidate #%d %s (%s) failed: %s", cand_idx + 1, date_str, scene_id, exc)
            finally:
                del ds, masked_ds, layer_da, img

        skipped_entry = {
            "date_str": prim_date_str,
            "scene_id": prim_scene_id,
            "valid_pct": round(first_fail_pct, 1),
            "reason": first_fail_reason,
        }
        return step, prim_date_str, None, None, skipped_entry

    if total > 0:
        batch_size = get_parallel_workers()
        completed = 0
        with ThreadPoolExecutor(max_workers=min(batch_size, total)) as executor:
            for batch_start in range(0, total, batch_size):
                batch = epoch_groups[batch_start : batch_start + batch_size]
                future_map = {
                    executor.submit(_process_tl_epoch, batch_start + offset, grp): (batch_start + offset, grp)
                    for offset, grp in enumerate(batch)
                }
                for fut in as_completed(future_map):
                    _step_idx, orig_grp = future_map[fut]
                    completed += 1
                    try:
                        _step, done_date, frame_entry, rep_entry, skip_entry = fut.result()
                        if frame_entry is not None:
                            frames.append(frame_entry)
                        if rep_entry is not None:
                            replaced.append(rep_entry)
                        if skip_entry is not None:
                            skipped.append(skip_entry)
                        if progress_callback is not None:
                            progress_callback(min(completed, total), total, done_date)
                    except Exception as exc:
                        _, err_date, err_id, _, _ = _group_metadata(orig_grp, sensor=sensor)
                        logger.warning("Timelapse worker failed for %s (%s): %s", err_date, err_id, exc)
                        skipped.append(
                            {
                                "date_str": err_date,
                                "scene_id": err_id,
                                "valid_pct": 0.0,
                                "reason": str(exc),
                            }
                        )
                        if progress_callback is not None:
                            progress_callback(min(completed, total), total, err_date)
                gc.collect()

    if progress_callback is not None:
        progress_callback(total, total, "done")

    frames.sort(key=lambda f: str(f.get("date_str", "")))
    skipped.sort(key=lambda d: str(d.get("date_str", "")))
    replaced.sort(key=lambda d: str(d.get("orig_date", "")))

    gif_bytes = (
        compile_timelapse_gif([f["png_bytes"] for f in frames], duration_ms=duration_ms)
        if frames
        else b""
    )
    gc.collect()

    return {
        "frames": frames,
        "gif_bytes": gif_bytes,
        "skipped": skipped,
        "replaced": replaced,
        "layer_mode": mode,
        "sensor": sensor,
    }


def _compute_aoi_clear_pct(
    ds: xr.Dataset,
    *,
    sensor: str,
    aoi_polygon_4326: dict[str, Any] | None = None,
) -> float:
    """Compute percentage [0..100] of genuinely cloud-free, non-NoData pixels inside the AOI."""
    data_vars = [
        str(v)
        for v in ds.data_vars
        if str(v) not in ("SCL", "QA_PIXEL", "qa_pixel")
    ]
    if not data_vars:
        return 0.0

    ref_da = ds[data_vars[0]]
    denom_px = int(ref_da.size)
    if aoi_polygon_4326 is not None and denom_px > 0:
        try:
            ones_da = xr.DataArray(
                np.ones(ref_da.shape, dtype=np.float32),
                dims=ref_da.dims,
                coords=ref_da.coords,
            )
            crs = ds.rio.crs or ref_da.rio.crs
            if crs is not None:
                ones_da = ones_da.rio.write_crs(crs, inplace=False)
            ones_da = ones_da.rio.write_nodata(np.nan, inplace=False)
            clipped_ones = clip_to_aoi_polygon(ones_da, aoi_polygon_4326, drop=False)
            poly_px = int(np.count_nonzero(np.isfinite(clipped_ones.values)))
            if poly_px > 0:
                denom_px = poly_px
        except Exception:
            pass

    if denom_px <= 0:
        return 0.0

    if sensor == SENSOR_S1:
        clear_mask = np.ones(ref_da.shape, dtype=bool)
        for v in data_vars:
            clear_mask &= np.isfinite(ds[v].values)
    else:
        masked_ds, _ = apply_cloud_mask(ds, mask_clouds=True, sensor=sensor)
        clear_mask = np.ones(ref_da.shape, dtype=bool)
        for v in data_vars:
            clear_mask &= np.isfinite(masked_ds[v].values)

    clear_px = int(np.count_nonzero(clear_mask))
    return round(float(np.clip(100.0 * clear_px / denom_px, 0.0, 100.0)), 1)


def load_clear_scene_with_fallback(
    target_date: str,
    all_candidates: Sequence[StacItemOrGroup],
    bbox_4326: Sequence[float],
    target_epsg: int | str = "EPSG:3794",
    sensor: str = SENSOR_S2,
    resolution: float = 10.0,
    min_clear_pct: float = 75.0,
    *,
    bands: Sequence[str] = ("B04", "B03", "B02"),
    primary_group: StacItemOrGroup | None = None,
    aoi_polygon_4326: dict[str, Any] | None = None,
    auto_fallback: bool = True,
    max_day_delta: int = 45,
    same_season_months: Sequence[int] | None = None,
    max_attempts: int = 6,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Load the target scene and verify actual cloud-free pixel share INSIDE the AOI.

    Uses Sentinel-2 ``SCL`` or Landsat ``QA_PIXEL`` to measure genuine cloud-free pixels over
    the AOI (eliminating cases where global STAC ``eo:cloud_cover`` is low but a cloud sits
    right on top of the AOI). If the chosen date has ``< min_clear_pct`` clear pixels and
    ``auto_fallback`` is enabled, automatically searches nearby candidate dates (preferring
    same-season / within ``±max_day_delta`` days and lowest cloud cover) until a genuinely
    clear scene is found.
    """
    is_sar = sensor == SENSOR_S1
    load_bands = [str(b) for b in bands]
    added_mask_band = False
    mask_band_name = "QA_PIXEL" if sensor == SENSOR_LANDSAT else "SCL"
    if (not is_sar) and (mask_band_name not in load_bands):
        load_bands.append(mask_band_name)
        added_mask_band = True

    # Normalize and deduplicate candidate groups by acquisition date
    norm_groups: list[list[pystac.Item]] = []
    seen_dates: set[str] = set()
    primary_norm: list[pystac.Item] | None = None
    if primary_group is not None:
        pg = [primary_group] if isinstance(primary_group, pystac.Item) else list(primary_group)
        if pg:
            primary_norm = pg
            _, p_date, _, _, _ = _group_metadata(pg, sensor=sensor)
            seen_dates.add(p_date)

    for raw_cand in all_candidates or ():
        grp = [raw_cand] if isinstance(raw_cand, pystac.Item) else [it for it in raw_cand if isinstance(it, pystac.Item)]
        if not grp:
            continue
        try:
            _, c_date, _, _, _ = _group_metadata(grp, sensor=sensor)
        except Exception:
            continue
        if primary_norm is None and c_date == str(target_date)[:10]:
            primary_norm = grp
            seen_dates.add(c_date)
            continue
        if c_date not in seen_dates:
            seen_dates.add(c_date)
            norm_groups.append(grp)

    if primary_norm is None:
        if not norm_groups:
            raise ValueError(f"No candidate STAC items available for target_date={target_date!r}.")
        primary_norm = norm_groups.pop(0)

    target_dt = date.fromisoformat(str(target_date)[:10])
    season_month_set = set(int(m) for m in same_season_months) if same_season_months else None

    def _fallback_sort_key(grp: list[pystac.Item]) -> tuple[int, int, int, float, str]:
        _, c_date, _, _, c_cld = _group_metadata(grp, sensor=sensor)
        c_dt = date.fromisoformat(c_date[:10])
        days_diff = abs((c_dt - target_dt).days)
        same_year = 0 if c_dt.year == target_dt.year else 1
        in_window = 0 if days_diff <= int(max_day_delta) else 1
        in_season = 0 if (season_month_set is None or c_dt.month in season_month_set) else 1
        cld_val = float(c_cld) if np.isfinite(c_cld) else 50.0
        # Prefer: same-year & in-season/in-window first, then closest date, then lowest cloud
        tier = 0 if (in_window == 0 and in_season == 0) else (
            1 if (same_year == 0 and in_season == 0) else (
                2 if in_window == 0 else (3 if same_year == 0 else 4)
            )
        )
        return (tier, same_year, days_diff, cld_val, c_date)

    fallback_groups = sorted(norm_groups, key=_fallback_sort_key)
    # Only consider fallbacks in the same year or within ±max_day_delta days (never jump years)
    fallback_groups = [
        g
        for g in fallback_groups
        if (
            abs((date.fromisoformat(_group_metadata(g, sensor=sensor)[1][:10]) - target_dt).days) <= max(int(max_day_delta), 120)
            or date.fromisoformat(_group_metadata(g, sensor=sensor)[1][:10]).year == target_dt.year
        )
    ]

    ordered_attempts = [primary_norm] + (fallback_groups[: max(0, int(max_attempts) - 1)] if auto_fallback else [])

    best_ds: xr.Dataset | None = None
    best_info: dict[str, Any] | None = None
    orig_clear_pct: float | None = None
    _, prim_date_str, _, _, _ = _group_metadata(primary_norm, sensor=sensor)

    for attempt_idx, grp in enumerate(ordered_attempts):
        _, d_str, s_id, plat, _ = _group_metadata(grp, sensor=sensor)
        try:
            ds_cand = load_scene(
                sensor=sensor,
                item=grp,
                bbox_4326=bbox_4326,
                bands=load_bands,
                target_epsg=target_epsg,
                resolution=float(resolution),
                aoi_polygon_4326=aoi_polygon_4326,
            )
        except Exception as exc:
            logger.warning("Failed loading candidate %s during cloud fallback: %s", d_str, exc)
            continue

        clear_pct = _compute_aoi_clear_pct(
            ds_cand, sensor=sensor, aoi_polygon_4326=aoi_polygon_4326
        )
        if attempt_idx == 0:
            orig_clear_pct = clear_pct

        if best_info is None or clear_pct > float(best_info["actual_clear_pct"]):
            if best_ds is not None:
                del best_ds
            best_ds = ds_cand
            best_info = {
                "requested_date": prim_date_str,
                "actual_date": d_str,
                "replaced": (attempt_idx > 0 and d_str != prim_date_str),
                "original_clear_pct": float(orig_clear_pct if orig_clear_pct is not None else clear_pct),
                "actual_clear_pct": float(clear_pct),
                "platform": plat,
                "scene_id": s_id,
                "item_group": grp,
            }
        else:
            del ds_cand

        if (not auto_fallback) or (clear_pct >= float(min_clear_pct)):
            break

    if best_ds is None or best_info is None:
        raise RuntimeError(f"Could not load any valid scene around {target_date}.")

    if added_mask_band and mask_band_name in best_ds.data_vars:
        best_ds = best_ds.drop_vars(mask_band_name)

    return best_ds, best_info


def render_multi_date_comparison(
    item_groups: Sequence[StacItemOrGroup],
    date_labels: Sequence[str],
    bbox_4326: Sequence[float],
    layer_mode: str = "RGB",
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
    mask_clouds: bool = True,
    max_width: int = 720,
    *,
    all_candidates: Sequence[StacItemOrGroup] | None = None,
    candidates_by_date: dict[str, list[list[pystac.Item]]] | None = None,
    auto_cloud_fallback: bool = True,
    min_clear_pct: float = 75.0,
    same_season_months: Sequence[int] | None = None,
    overlay_max_width: int = 1800,
) -> dict[str, Any]:
    """Render 2 or 3 pixel-aligned dates (RGB, NIR, or Spectral Index / SAR) for multi-date comparison.

    Performs automatic pixel-level AOI cloud-free validation & fallback via
    ``load_clear_scene_with_fallback`` (default ``min_clear_pct=75.0``) so cloudy dates
    over the AOI are swapped with the nearest genuinely clear scene.

    Returns identical-dimension ``swipe_pngs``, ``strip_pngs``, high-resolution EPSG:4326
    Leaflet ``overlay_uris`` + ``overlay_bounds`` (for the zoomable/pannable curtain swipe map),
    ``replacements`` log, and when an index/SAR metric is selected, ``da_first``, ``da_last``,
    and ``diff_da = da_last - da_first``.
    """
    mode = layer_mode.strip().upper()
    is_sar = sensor == SENSOR_S1
    is_rgb_composite = mode in ("RGB", "NIR") and not is_sar

    if is_sar:
        if mode not in SAR_METRICS:
            mode = "VV"
        req_bands = list(SAR_METRICS[mode]["pols"])
        use_cloud_mask = False
    elif mode == "RGB":
        req_bands = ["B04", "B03", "B02"]
        use_cloud_mask = False
    elif mode == "NIR":
        req_bands = ["B08", "B04", "B03"]
        use_cloud_mask = False
    else:
        use_cloud_mask = bool(mask_clouds)
        req_bands = get_required_bands_for_sensor(sensor, mode, include_cloud_mask=use_cloud_mask)

    swipe_pngs: list[bytes] = []
    strip_pngs: list[bytes] = []
    overlay_uris: list[str] = []
    overlay_bounds: list[list[list[float]]] = []
    actual_dates: list[str] = []
    clear_pcts: list[float] = []
    replacements: list[dict[str, Any]] = []
    target_size: tuple[int, int] | None = None
    da_first: xr.DataArray | None = None
    da_last: xr.DataArray | None = None

    for idx, (grp, d_lbl) in enumerate(zip(item_groups, date_labels)):
        cand_pool: list[StacItemOrGroup] = []
        if candidates_by_date and str(d_lbl) in candidates_by_date:
            cand_pool.extend(candidates_by_date[str(d_lbl)])
        if all_candidates:
            cand_pool.extend(all_candidates)

        ds, fb_info = load_clear_scene_with_fallback(
            target_date=str(d_lbl),
            all_candidates=cand_pool,
            bbox_4326=bbox_4326,
            target_epsg=target_epsg,
            sensor=sensor,
            resolution=float(resolution),
            min_clear_pct=float(min_clear_pct),
            bands=req_bands,
            primary_group=grp,
            aoi_polygon_4326=aoi_polygon_4326,
            auto_fallback=bool(auto_cloud_fallback),
            same_season_months=same_season_months,
        )
        eff_date = str(fb_info["actual_date"])
        actual_dates.append(eff_date)
        clear_pcts.append(float(fb_info["actual_clear_pct"]))
        if fb_info.get("replaced"):
            replacements.append(
                {
                    "requested_date": str(fb_info["requested_date"]),
                    "actual_date": eff_date,
                    "original_clear_pct": float(fb_info["original_clear_pct"]),
                    "actual_clear_pct": float(fb_info["actual_clear_pct"]),
                }
            )

        try:
            if is_rgb_composite:
                y_coords = np.asarray(ds.coords["y"].values, dtype=np.float64)
                rgb_u8 = _stretch_rgb_for_pil(ds, req_bands, y_coords)
                im = Image.fromarray(rgb_u8, mode="RGB")
                im = _resize_pil_frame(im, max_width=max_width, target_size=target_size)
                if target_size is None:
                    target_size = im.size
                swipe_pngs.append(pil_to_png_bytes(im))
                strip_pngs.append(pil_to_png_bytes(_stamp_banner_on_pil(im.copy(), f"{eff_date} | {mode}")))

                ov_uri, ov_bnds = raster_to_map_overlay_png(
                    ds,
                    src_epsg=target_epsg,
                    is_rgb=True,
                    bands=req_bands,
                    use_percentile_stretch=True,
                    gamma=1.0,
                    max_width=int(overlay_max_width),
                )
                overlay_uris.append(ov_uri)
                overlay_bounds.append(ov_bnds)
            else:
                masked_ds, _ = apply_cloud_mask(ds, mask_clouds=use_cloud_mask, sensor=sensor)
                layer_da = _compute_layer(masked_ds, sensor=sensor, metric_name=mode)
                if idx == 0:
                    da_first = layer_da
                if idx == len(item_groups) - 1:
                    da_last = layer_da
                spec = SAR_METRICS[mode] if is_sar else SPECTRAL_INDICES[mode]
                im = render_dataarray_to_pil(
                    layer_da,
                    cmap_name=str(spec["cmap"]),
                    vmin=float(spec["vmin"]),
                    vmax=float(spec["vmax"]),
                    stamp_text=None,
                    max_width=max_width,
                    target_size=target_size,
                )
                if target_size is None:
                    target_size = im.size
                swipe_pngs.append(pil_to_png_bytes(im))
                strip_pngs.append(pil_to_png_bytes(_stamp_banner_on_pil(im.copy(), f"{eff_date} | {mode}")))

                ov_uri, ov_bnds = raster_to_map_overlay_png(
                    layer_da,
                    src_epsg=target_epsg,
                    is_rgb=False,
                    cmap_name=str(spec["cmap"]),
                    vmin=float(spec["vmin"]),
                    vmax=float(spec["vmax"]),
                    max_width=int(overlay_max_width),
                )
                overlay_uris.append(ov_uri)
                overlay_bounds.append(ov_bnds)
        finally:
            del ds
            gc.collect()

    diff_da: xr.DataArray | None = None
    if da_first is not None and da_last is not None:
        if not _grids_match(da_first, da_last):
            da_last = da_last.rio.reproject_match(da_first, resampling=Resampling.bilinear)
        diff_vals = (da_last.values - da_first.values).astype(np.float32)
        diff_da = xr.DataArray(
            diff_vals,
            dims=da_first.dims,
            coords=da_first.coords,
            name=f"Delta_{mode}",
        )
        if da_first.rio.crs is not None:
            diff_da = diff_da.rio.write_crs(da_first.rio.crs, inplace=False)
        diff_da = diff_da.rio.write_nodata(np.nan, inplace=False)

    return {
        "dates": actual_dates,
        "requested_dates": list(date_labels),
        "clear_pcts": clear_pcts,
        "replacements": replacements,
        "layer_mode": mode,
        "swipe_pngs": swipe_pngs,
        "strip_pngs": strip_pngs,
        "overlay_uris": overlay_uris,
        "overlay_bounds": overlay_bounds,
        "bbox_4326": [float(v) for v in bbox_4326],
        "da_first": da_first,
        "da_last": da_last,
        "diff_da": diff_da,
    }


def build_multidate_visual_frames(
    item_groups: Sequence[StacItemOrGroup],
    date_labels: Sequence[str],
    bbox_4326: Sequence[float],
    index_name: str = "NDVI",
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 20,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
    *,
    extra_indices: Sequence[str] = (),
    max_frames: int = 9,
    all_candidates: Sequence[StacItemOrGroup] | None = None,
    candidates_by_date: dict[str, list[list[pystac.Item]]] | None = None,
    auto_cloud_fallback: bool = True,
    min_clear_pct: float = 75.0,
    max_thumb_width: int = 640,
    overlay_max_width: int = 1200,
    default_fps: float = 1.5,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Load an evenly-spaced chronological stack of clear intermediate visual frames (RGB + multi-index).

    Streams each epoch ONCE via ``load_clear_scene_with_fallback`` (loading all bands for RGB,
    ``index_name``, and ``extra_indices`` together), encodes compact Leaflet overlays + date-stamped
    PNG frames, immediately frees raw xarrays (``gc.collect()``), computes the diverging spatial
    change map (``Δ = t2 - t1``), and pre-compiles RGB and Index Timelapse GIFs at ``default_fps``.
    """
    n_in = min(len(item_groups), len(date_labels))
    if n_in == 0:
        return {
            "frames": [],
            "timelapse_rgb_gif_bytes": None,
            "timelapse_index_gif_bytes": None,
            "da_first": None,
            "da_last": None,
            "diff_da": None,
            "delta_overlay_uri": None,
            "delta_overlay_bounds": None,
            "delta_vmin": -0.5,
            "delta_vmax": 0.5,
            "delta_cmap": "RdBu",
            "replacements": [],
        }

    # Sort chronologically oldest -> newest
    order = sorted(range(n_in), key=lambda i: str(date_labels[i]))
    sorted_groups = [item_groups[i] for i in order]
    sorted_dates = [str(date_labels[i]) for i in order]

    # Evenly sample up to max_frames while strictly preserving earliest & latest dates
    cap = max(2, int(max_frames))
    if len(sorted_groups) > cap:
        pick_idx = sorted(set(int(v) for v in np.linspace(0, len(sorted_groups) - 1, cap, dtype=int)))
        sorted_groups = [sorted_groups[i] for i in pick_idx]
        sorted_dates = [sorted_dates[i] for i in pick_idx]

    is_sar = sensor == SENSOR_S1
    idx_key = str(index_name or ("VV" if is_sar else "NDVI")).strip().upper()
    clean_extras: list[str] = []
    if is_sar:
        if idx_key not in SAR_METRICS:
            idx_key = "VV"
        req_bands = list(SAR_METRICS[idx_key]["pols"])
        idx_spec = SAR_METRICS[idx_key]
    else:
        if idx_key not in SPECTRAL_INDICES:
            idx_key = "NDVI"
        idx_bands = get_required_bands_for_sensor(sensor, idx_key, include_cloud_mask=True)
        extra_band_list: list[str] = []
        for ex in extra_indices:
            ex_u = str(ex or "").strip().upper()
            if not ex_u or ex_u in ("RGB", "SAR", idx_key) or ex_u in clean_extras:
                continue
            if ex_u == "SWIR":
                clean_extras.append("SWIR")
                extra_band_list.extend(["B12", "B08", "B04"])
            elif ex_u in SPECTRAL_INDICES:
                clean_extras.append(ex_u)
                extra_band_list.extend(get_required_bands_for_sensor(sensor, ex_u, include_cloud_mask=False))
        # Merge RGB + primary index + complementary indices + cloud mask into ONE STAC load per epoch
        req_bands = list(dict.fromkeys(["B04", "B03", "B02", *idx_bands, *extra_band_list]))
        idx_spec = SPECTRAL_INDICES[idx_key]

    frames: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    seen_actual_dates: set[str] = set()
    target_size: tuple[int, int] | None = None
    da_first: xr.DataArray | None = None
    da_last: xr.DataArray | None = None

    total = len(sorted_groups)
    for idx_i, (grp, d_lbl) in enumerate(zip(sorted_groups, sorted_dates)):
        if progress_callback is not None:
            progress_callback(idx_i + 1, total, d_lbl)

        cand_pool: list[StacItemOrGroup] = []
        if candidates_by_date and str(d_lbl) in candidates_by_date:
            cand_pool.extend(candidates_by_date[str(d_lbl)])
        if all_candidates:
            cand_pool.extend(all_candidates)
        if not cand_pool:
            cand_pool = [grp]

        try:
            ds_ep, fb_info = load_clear_scene_with_fallback(
                target_date=str(d_lbl),
                all_candidates=cand_pool,
                bbox_4326=bbox_4326,
                target_epsg=target_epsg,
                sensor=sensor,
                resolution=float(resolution),
                min_clear_pct=float(min_clear_pct),
                bands=req_bands,
                primary_group=grp,
                aoi_polygon_4326=aoi_polygon_4326,
                auto_fallback=bool(auto_cloud_fallback and not is_sar),
            )
            act_date = str(fb_info.get("actual_date") or d_lbl)
            clear_pct = float(fb_info.get("actual_clear_pct", 100.0))
            rep_info = fb_info if fb_info.get("replaced") else None
        except Exception as exc:
            logger.warning("Skipping visual frame %s: %s", d_lbl, exc)
            continue

        if act_date in seen_actual_dates:
            del ds_ep
            gc.collect()
            continue
        seen_actual_dates.add(act_date)

        if rep_info is not None:
            replacements.append(rep_info)

        plat_raw = str(ds_ep.attrs.get("platform") or fb_info.get("platform") or "")
        plat_lbl = _format_platform_stamp(plat_raw, sensor)
        res_int = int(round(float(resolution)))
        plat_banner = f"{plat_lbl} ({res_int} m)"
        y_coords = np.asarray(ds_ep.coords["y"].values, dtype=np.float64)

        # 1. True Color RGB overlay & date-stamped PNG (always for optical sensors)
        rgb_uri: str | None = None
        rgb_png: bytes | None = None
        bounds_4326: list[list[float]] | None = None

        if not is_sar and all(b in ds_ep.data_vars for b in ("B04", "B03", "B02")):
            try:
                rgb_uri, bounds_4326 = raster_to_map_overlay_png(
                    ds_ep,
                    src_epsg=target_epsg,
                    is_rgb=True,
                    bands=["B04", "B03", "B02"],
                    max_width=overlay_max_width,
                )
                rgb_u8 = _stretch_rgb_for_pil(ds_ep, ["B04", "B03", "B02"], y_coords)
                pil_rgb = Image.fromarray(rgb_u8, mode="RGB")
                pil_rgb = _resize_pil_frame(pil_rgb, max_width=max_thumb_width, target_size=target_size)
                if target_size is None:
                    target_size = pil_rgb.size
                pil_rgb_stamped = _stamp_banner_on_pil(pil_rgb, f"{act_date} | {plat_banner} | RGB")
                rgb_png = pil_to_png_bytes(pil_rgb_stamped)
                del rgb_u8, pil_rgb, pil_rgb_stamped
            except Exception as exc:
                logger.warning("RGB frame render skipped for %s: %s", act_date, exc)

        # Cloud-masked dataset for clean spectral indices (MODIS 8-day composites are pre-composited)
        if not is_sar and sensor not in MODIS_SENSORS:
            try:
                ds_masked, _ = apply_cloud_mask(ds_ep, mask_clouds=True, sensor=sensor)
            except Exception:
                ds_masked = ds_ep
        else:
            ds_masked = ds_ep

        # 2. Primary Spectral Index / SAR overlay, scalar mean, lightweight grid & date-stamped PNG
        idx_uri: str | None = None
        idx_png: bytes | None = None
        idx_mean: float | None = None
        index_grid_small: dict[str, Any] | None = None
        ndwi_grid_small: dict[str, Any] | None = None
        nir_grid_small: dict[str, Any] | None = None
        try:
            da_idx = _compute_layer(ds_masked, sensor=sensor, metric_name=idx_key)
            vals_arr = np.asarray(da_idx.values, dtype=np.float32)
            fin = vals_arr[np.isfinite(vals_arr)]
            if fin.size > 0:
                idx_mean = round(float(np.mean(fin)), 4)

            try:
                index_grid_small = _downsample_da_to_small_grid(da_idx, max_dim=280, fallback_crs=target_epsg)
                if idx_key == "NDSI" and not is_sar:
                    if "B08" in ds_masked.data_vars:
                        nir_grid_small = _downsample_da_to_small_grid(ds_masked["B08"], max_dim=280, fallback_crs=target_epsg)
                    if all(b in ds_masked.data_vars for b in ("B03", "B08")):
                        da_ndwi_tmp = _compute_layer(ds_masked, sensor=sensor, metric_name="NDWI")
                        ndwi_grid_small = _downsample_da_to_small_grid(da_ndwi_tmp, max_dim=280, fallback_crs=target_epsg)
                        del da_ndwi_tmp
            except Exception:
                pass

            vmin_v = float(idx_spec["vmin"])
            vmax_v = float(idx_spec["vmax"])
            cmap_n = str(idx_spec["cmap"])

            idx_uri, idx_bnds = raster_to_map_overlay_png(
                da_idx,
                src_epsg=target_epsg,
                is_rgb=False,
                cmap_name=cmap_n,
                vmin=vmin_v,
                vmax=vmax_v,
                max_width=overlay_max_width,
            )
            if bounds_4326 is None:
                bounds_4326 = idx_bnds

            mean_tag = f" | {idx_key}={idx_mean:.2f}" if idx_mean is not None else f" | {idx_key}"
            pil_idx = render_dataarray_to_pil(
                da_idx,
                cmap_name=cmap_n,
                vmin=vmin_v,
                vmax=vmax_v,
                stamp_text=f"{act_date} | {plat_banner}{mean_tag}",
                max_width=max_thumb_width,
                target_size=target_size,
            )
            if target_size is None:
                target_size = pil_idx.size
            idx_png = pil_to_png_bytes(pil_idx)
            del pil_idx

            if da_first is None:
                da_first = da_idx
            da_last = da_idx
        except Exception as exc:
            logger.warning("Index frame render skipped for %s: %s", act_date, exc)

        # 3. Complementary Spectral Indices / SWIR False Color from the SAME loaded scene!
        extra_overlays: dict[str, str] = {}
        extra_pngs: dict[str, bytes] = {}
        extra_means: dict[str, float | None] = {}
        for ex_code in clean_extras:
            try:
                if ex_code == "SWIR":
                    if all(b in ds_ep.data_vars for b in ("B12", "B08", "B04")):
                        sw_uri, sw_bnds = raster_to_map_overlay_png(
                            ds_ep,
                            src_epsg=target_epsg,
                            is_rgb=True,
                            bands=["B12", "B08", "B04"],
                            max_width=overlay_max_width,
                        )
                        if bounds_4326 is None:
                            bounds_4326 = sw_bnds
                        extra_overlays["SWIR"] = sw_uri
                        sw_u8 = _stretch_rgb_for_pil(ds_ep, ["B12", "B08", "B04"], y_coords)
                        pil_sw = _resize_pil_frame(
                            Image.fromarray(sw_u8, mode="RGB"),
                            max_width=max_thumb_width,
                            target_size=target_size,
                        )
                        extra_pngs["SWIR"] = pil_to_png_bytes(
                            _stamp_banner_on_pil(pil_sw, f"{act_date} | {plat_lbl} | SWIR")
                        )
                        extra_means["SWIR"] = None
                        del sw_u8, pil_sw
                elif ex_code in SPECTRAL_INDICES:
                    da_ex = _compute_layer(ds_masked, sensor=sensor, metric_name=ex_code)
                    ex_vals = np.asarray(da_ex.values, dtype=np.float32)
                    ex_fin = ex_vals[np.isfinite(ex_vals)]
                    ex_mean = round(float(np.mean(ex_fin)), 4) if ex_fin.size > 0 else None
                    ex_spec = SPECTRAL_INDICES[ex_code]
                    ex_uri, ex_bnds = raster_to_map_overlay_png(
                        da_ex,
                        src_epsg=target_epsg,
                        is_rgb=False,
                        cmap_name=str(ex_spec["cmap"]),
                        vmin=float(ex_spec["vmin"]),
                        vmax=float(ex_spec["vmax"]),
                        max_width=overlay_max_width,
                    )
                    if bounds_4326 is None:
                        bounds_4326 = ex_bnds
                    extra_overlays[ex_code] = ex_uri
                    extra_means[ex_code] = ex_mean
                    ex_tag = f" | {ex_code}={ex_mean:.2f}" if ex_mean is not None else f" | {ex_code}"
                    pil_ex = render_dataarray_to_pil(
                        da_ex,
                        cmap_name=str(ex_spec["cmap"]),
                        vmin=float(ex_spec["vmin"]),
                        vmax=float(ex_spec["vmax"]),
                        stamp_text=f"{act_date} | {plat_lbl}{ex_tag}",
                        max_width=max_thumb_width,
                        target_size=target_size,
                    )
                    extra_pngs[ex_code] = pil_to_png_bytes(pil_ex)
                    del da_ex, pil_ex
            except Exception as exc:
                logger.warning("Complementary layer %s skipped for %s: %s", ex_code, act_date, exc)

        sat_pct_fr = compute_optical_saturation_pct(ds_masked, sensor=sensor)
        if ds_masked is not ds_ep:
            del ds_masked
        del ds_ep
        gc.collect()

        if rgb_uri or idx_uri:
            if sensor in MODIS_SENSORS:
                sensor_fam = "MODIS (500 m–1 km)"
            elif sensor == SENSOR_LANDSAT:
                sensor_fam = "Landsat (30 m)"
            elif sensor == SENSOR_S1:
                sensor_fam = "Sentinel-1 SAR"
            else:
                sensor_fam = "Sentinel-2 (10–20 m)"
            rel_ok_fr, q_note_fr = assess_scene_analysis_reliability(
                act_date,
                sensor_fam,
                float(clear_pct),
                saturated_pct=sat_pct_fr,
            )
            frames.append(
                {
                    "date": act_date,
                    "requested_date": str(d_lbl),
                    "sensor": sensor,
                    "sensor_family": sensor_fam,
                    "platform": plat_lbl,
                    "resolution_m": res_int,
                    "clear_pct": round(float(clear_pct), 1),
                    "saturated_pct": sat_pct_fr,
                    "analysis_reliable": rel_ok_fr,
                    "quality_note": q_note_fr,
                    "index_name": idx_key,
                    "index_mean": idx_mean,
                    "bounds_4326": bounds_4326,
                    "rgb_overlay_uri": rgb_uri,
                    "index_overlay_uri": idx_uri,
                    "extra_overlays": extra_overlays,
                    "extra_pngs": extra_pngs,
                    "extra_means": extra_means,
                    "rgb_png_bytes": rgb_png,
                    "index_png_bytes": idx_png,
                    "index_grid_small": index_grid_small,
                    "ndwi_grid_small": ndwi_grid_small,
                    "nir_grid_small": nir_grid_small,
                }
            )

    # Pixel-aligned Δ between earliest RADIOMETRICALLY RELIABLE (>=1990) and latest clear frames (water-aware for NDSI!)
    default_t1_idx = get_default_reliable_t1_index(frames)
    if len(frames) >= 2 and default_t1_idx >= len(frames) - 1:
        default_t1_idx = 0

    diff_da: xr.DataArray | None = None
    delta_uri: str | None = None
    delta_bnds: list[list[float]] | None = None
    delta_vmin: float = -0.6
    delta_vmax: float = 0.6
    delta_cmap: str = "RdBu"
    glacier_class_uri: str | None = None
    glacier_class_bnds: list[list[float]] | None = None
    glacier_change_stats: dict[str, float] | None = None

    if len(frames) >= 2:
        try:
            dyn_init = compute_dynamic_frame_delta(
                frames[default_t1_idx],
                frames[-1],
                target_epsg=target_epsg,
                index_name=idx_key,
                max_width=overlay_max_width,
            )
            if isinstance(dyn_init, dict):
                diff_da = dyn_init["diff_da"]
                delta_uri = dyn_init["delta_overlay_uri"]
                delta_bnds = dyn_init["delta_overlay_bounds"]
                delta_vmin = dyn_init["delta_vmin"]
                delta_vmax = dyn_init["delta_vmax"]
                delta_cmap = dyn_init["delta_cmap"]
                glacier_class_uri = dyn_init.get("glacier_class_uri")
                glacier_class_bnds = dyn_init.get("glacier_class_bounds")
                glacier_change_stats = dyn_init.get("glacier_change_stats")
        except Exception as exc:
            logger.warning("Dynamic Δ change overlay skipped: %s", exc)

    if diff_da is None and da_first is not None and da_last is not None and len(frames) >= 2:
        try:
            da_l_aligned = da_last
            if not _grids_match(da_first, da_l_aligned):
                da_l_aligned = da_l_aligned.rio.reproject_match(da_first, resampling=Resampling.bilinear)
            diff_vals = (da_l_aligned.values - da_first.values).astype(np.float32)
            diff_da = xr.DataArray(
                diff_vals,
                dims=da_first.dims,
                coords=da_first.coords,
                name=f"Delta_{idx_key}",
            )
            if da_first.rio.crs is not None:
                diff_da = diff_da.rio.write_crs(da_first.rio.crs, inplace=False)
            diff_da = diff_da.rio.write_nodata(np.nan, inplace=False)
            delta_uri, delta_bnds, delta_vmin, delta_vmax, delta_cmap = render_diverging_change_overlay(
                diff_da,
                src_epsg=target_epsg,
                index_name=idx_key,
                max_width=overlay_max_width,
            )
        except Exception as exc:
            logger.warning("Diverging Δ change overlay skipped: %s", exc)

    # Pre-compile default RGB and Index Timelapse GIFs at default_fps
    rgb_png_list = [f["rgb_png_bytes"] for f in frames if f.get("rgb_png_bytes")]
    idx_png_list = [f["index_png_bytes"] for f in frames if f.get("index_png_bytes")]
    gif_rgb = compile_timelapse_gif_fps(rgb_png_list, fps=default_fps) if len(rgb_png_list) >= 2 else None
    gif_idx = compile_timelapse_gif_fps(idx_png_list, fps=default_fps) if len(idx_png_list) >= 2 else None

    gc.collect()
    return {
        "frames": frames,
        "default_t1_index": default_t1_idx,
        "timelapse_rgb_gif_bytes": gif_rgb,
        "timelapse_index_gif_bytes": gif_idx,
        "da_first": da_first,
        "da_last": da_last,
        "diff_da": diff_da,
        "delta_overlay_uri": delta_uri,
        "delta_overlay_bounds": delta_bnds,
        "delta_vmin": delta_vmin,
        "delta_vmax": delta_vmax,
        "delta_cmap": delta_cmap,
        "glacier_class_uri": glacier_class_uri,
        "glacier_class_bounds": glacier_class_bnds,
        "glacier_change_stats": glacier_change_stats,
        "replacements": replacements,
    }


# ---------------------------------------------------------------------------
# Cloud-Free Temporal Median Composite (Memory-Safe Sequential Streaming)
# ---------------------------------------------------------------------------
COMPOSITE_RGB_NIR_BANDS: tuple[str, ...] = ("B04", "B03", "B02", "B08")


def _select_least_cloudy_groups(
    items: Sequence[StacItemOrGroup],
    sensor: str,
    max_scenes: int,
) -> list[list[pystac.Item]]:
    """Select up to ``max_scenes`` least-cloudy unique-date STAC groups, returned chronologically."""
    groups = _normalize_epoch_groups(items)
    if len(groups) <= max_scenes:
        return groups

    scored: list[tuple[float, str, list[pystac.Item]]] = []
    for day_items in groups:
        _, date_str, _, _, cloud = _group_metadata(day_items, sensor=sensor)
        sort_cloud = cloud if np.isfinite(cloud) else 999.0
        scored.append((sort_cloud, date_str, day_items))

    scored.sort(key=lambda tup: (tup[0], tup[1]))
    chosen = scored[: int(max_scenes)]
    chosen.sort(key=lambda tup: tup[1])
    return [g for _, _, g in chosen]


def compute_cloud_free_composite(
    items: Sequence[StacItemOrGroup],
    bbox_4326: Sequence[float],
    mode: str = "RGB_NIR",
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
    max_scenes: int = 6,
    min_valid_pct: float = 5.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[xr.Dataset | xr.DataArray | None, dict[str, Any]]:
    """Compute a pixel-wise cloud-free median composite across up to ``max_scenes`` clearest dates.

    Streams up to 4 same-day mosaicked acquisitions in parallel via
    ``ThreadPoolExecutor(max_workers=4)`` with batch ``gc.collect()`` (16 GB RAM safe),
    applies Sentinel-2 SCL or Landsat ``QA_PIXEL`` cloud/shadow masking, aligns
    every epoch to the reference pixel grid, and computes the per-pixel ``nanmedian``.

    Parameters
    ----------
    mode : str
        ``"RGB_NIR"`` (loads ``B04, B03, B02, B08`` for True Color RGB & False Color NIR)
        or an optical spectral index key (``"NDVI"``, ``"NDWI"``, ``"MNDWI"``, ``"NDBI"``, ``"NBR"``).
    """
    if sensor == SENSOR_S1:
        raise ValueError("Cloud-free optical median composite applies to optical sensors (Sentinel-2 / Landsat).")

    mode_key = mode.strip().upper()
    is_rgb_nir = mode_key in ("RGB_NIR", "RGB", "NIR")
    if is_rgb_nir:
        mode_key = "RGB_NIR"
        mask_band = "QA_PIXEL" if sensor == SENSOR_LANDSAT else "SCL"
        req_bands = [*COMPOSITE_RGB_NIR_BANDS, mask_band]
    else:
        if mode_key not in SPECTRAL_INDICES:
            raise ValueError(
                f"Unsupported composite mode {mode!r}. Use 'RGB_NIR' or one of {list(SPECTRAL_INDICES.keys())}."
            )
        req_bands = get_required_bands_for_sensor(sensor, mode_key, include_cloud_mask=True)

    selected_groups = _select_least_cloudy_groups(items, sensor=sensor, max_scenes=max(1, int(max_scenes)))
    total = len(selected_groups)

    ref_da: xr.DataArray | None = None
    ref_crs = None
    band_stacks: dict[str, list[np.ndarray]] = (
        {b: [] for b in COMPOSITE_RGB_NIR_BANDS} if is_rgb_nir else {mode_key: []}
    )
    dates_used: list[str] = []
    skipped_dates: list[dict[str, Any]] = []

    def _load_comp_epoch(
        step: int,
        day_items: list[pystac.Item],
    ) -> tuple[int, str, str, xr.Dataset | xr.DataArray | None, dict[str, Any] | None]:
        _, date_str, scene_id, _, _ = _group_metadata(day_items, sensor=sensor)
        ds: xr.Dataset | None = None
        masked_ds: xr.Dataset | None = None
        try:
            ds = load_scene(
                sensor=sensor,
                item=day_items,
                bbox_4326=bbox_4326,
                bands=req_bands,
                target_epsg=target_epsg,
                resolution=float(resolution),
                aoi_polygon_4326=aoi_polygon_4326,
            )
            masked_ds, clear_pct = apply_cloud_mask(ds, mask_clouds=True, sensor=sensor)
            if clear_pct < float(min_valid_pct):
                return (
                    step,
                    date_str,
                    scene_id,
                    None,
                    {
                        "date_str": date_str,
                        "scene_id": scene_id,
                        "valid_pct": round(clear_pct, 1),
                        "reason": "cloudy",
                    },
                )
            if is_rgb_nir:
                keep_ds = masked_ds[list(COMPOSITE_RGB_NIR_BANDS)]
                if masked_ds.rio.crs is not None:
                    keep_ds = keep_ds.rio.write_crs(masked_ds.rio.crs, inplace=False)
                return step, date_str, scene_id, keep_ds, None
            layer_da = compute_index(masked_ds, mode_key)
            return step, date_str, scene_id, layer_da, None
        except Exception as exc:
            logger.warning("Composite skipped %s (%s): %s", date_str, scene_id, exc)
            return (
                step,
                date_str,
                scene_id,
                None,
                {"date_str": date_str, "scene_id": scene_id, "valid_pct": 0.0, "reason": str(exc)},
            )
        finally:
            del ds, masked_ds

    if total > 0:
        batch_size = get_parallel_workers()
        completed = 0
        with ThreadPoolExecutor(max_workers=min(batch_size, total)) as executor:
            for batch_start in range(0, total, batch_size):
                batch = selected_groups[batch_start : batch_start + batch_size]
                future_map = {
                    executor.submit(_load_comp_epoch, batch_start + offset, grp): (batch_start + offset, grp)
                    for offset, grp in enumerate(batch)
                }
                batch_results: list[
                    tuple[int, str, str, xr.Dataset | xr.DataArray | None, dict[str, Any] | None]
                ] = []
                for fut in as_completed(future_map):
                    step_idx, orig_grp = future_map[fut]
                    completed += 1
                    try:
                        res = fut.result()
                        batch_results.append(res)
                        if progress_callback is not None:
                            progress_callback(min(completed, total), total, res[1])
                    except Exception as exc:
                        _, err_date, err_id, _, _ = _group_metadata(orig_grp, sensor=sensor)
                        logger.warning("Composite worker failed %s (%s): %s", err_date, err_id, exc)
                        batch_results.append(
                            (
                                step_idx,
                                err_date,
                                err_id,
                                None,
                                {"date_str": err_date, "scene_id": err_id, "valid_pct": 0.0, "reason": str(exc)},
                            )
                        )
                        if progress_callback is not None:
                            progress_callback(min(completed, total), total, err_date)

                # Process completed batch in chronological order so ref_da is deterministic
                batch_results.sort(key=lambda item: item[0])
                for _step, date_str, scene_id, payload, skip_entry in batch_results:
                    if skip_entry is not None or payload is None:
                        if skip_entry is not None:
                            skipped_dates.append(skip_entry)
                        continue
                    try:
                        if is_rgb_nir:
                            assert isinstance(payload, xr.Dataset)
                            if ref_da is None:
                                ref_da = payload[COMPOSITE_RGB_NIR_BANDS[0]].copy()
                                ref_crs = payload.rio.crs or ref_da.rio.crs
                            for b in COMPOSITE_RGB_NIR_BANDS:
                                da_b = payload[b]
                                if not _grids_match(ref_da, da_b):
                                    da_b = da_b.rio.reproject_match(
                                        ref_da, resampling=Resampling.bilinear, nodata=np.nan
                                    )
                                band_stacks[b].append(da_b.values.astype(np.float32, copy=True))
                        else:
                            assert isinstance(payload, xr.DataArray)
                            layer_da = payload
                            if ref_da is None:
                                ref_da = layer_da.copy()
                                ref_crs = layer_da.rio.crs
                            elif not _grids_match(ref_da, layer_da):
                                layer_da = layer_da.rio.reproject_match(
                                    ref_da, resampling=Resampling.bilinear, nodata=np.nan
                                )
                            band_stacks[mode_key].append(layer_da.values.astype(np.float32, copy=True))
                        dates_used.append(date_str)
                    except Exception as exc:
                        logger.warning("Composite alignment skipped %s (%s): %s", date_str, scene_id, exc)
                        skipped_dates.append(
                            {"date_str": date_str, "scene_id": scene_id, "valid_pct": 0.0, "reason": str(exc)}
                        )
                del batch_results
                gc.collect()

    if progress_callback is not None:
        progress_callback(total, total, "done")

    dates_used.sort()
    skipped_dates.sort(key=lambda d: str(d.get("date_str", "")))

    if not dates_used or ref_da is None:
        return None, {
            "mode": mode_key,
            "sensor": sensor,
            "scenes_requested": total,
            "scenes_used": 0,
            "dates_used": [],
            "skipped_dates": skipped_dates,
            "date_start": "",
            "date_end": "",
            "valid_pct": 0.0,
            "valid_area_ha": 0.0,
            "median_obs_per_px": 0.0,
            "resolution_m": float(resolution),
            "crs": str(target_epsg),
        }

    date_start = min(dates_used)
    date_end = max(dates_used)
    px_area_ha = (float(resolution) ** 2) / 10_000.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if is_rgb_nir:
            med_vars: dict[str, xr.DataArray] = {}
            obs_counts = None
            for b in COMPOSITE_RGB_NIR_BANDS:
                stack = np.stack(band_stacks[b], axis=0)
                if obs_counts is None:
                    obs_counts = np.sum(np.isfinite(stack), axis=0)
                med = np.nanmedian(stack, axis=0).astype(np.float32)
                del stack
                band_stacks[b].clear()

                da_med = xr.DataArray(
                    med,
                    dims=ref_da.dims,
                    coords=ref_da.coords,
                    name=b,
                    attrs={
                        "units": "surface_reflectance",
                        "long_name": f"Cloud-Free Median {b} ({date_start} to {date_end})",
                    },
                )
                if ref_crs is not None:
                    da_med = da_med.rio.write_crs(ref_crs, inplace=False)
                da_med = da_med.rio.write_nodata(np.nan, inplace=False)
                med_vars[b] = da_med

            valid_mask = np.ones(ref_da.shape, dtype=bool)
            for b in COMPOSITE_RGB_NIR_BANDS:
                valid_mask &= np.isfinite(med_vars[b].values)

            valid_pixels = int(np.count_nonzero(valid_mask))
            total_pixels = int(valid_mask.size)
            valid_pct = round(100.0 * valid_pixels / max(total_pixels, 1), 2)
            valid_area_ha = round(valid_pixels * px_area_ha, 3)
            med_obs = float(np.median(obs_counts[valid_mask])) if (obs_counts is not None and valid_pixels > 0) else 0.0

            result_obj: xr.Dataset | xr.DataArray = xr.Dataset(
                data_vars=med_vars,
                coords=ref_da.coords,
                attrs={
                    "sensor": sensor,
                    "composite_mode": mode_key,
                    "datetime": f"{date_start}..{date_end}",
                    "date_start": date_start,
                    "date_end": date_end,
                    "scenes_used": len(dates_used),
                    "dates_used": ", ".join(dates_used),
                    "resolution_m": float(resolution),
                    "crs": str(ref_crs) if ref_crs is not None else str(target_epsg),
                },
            )
            if ref_crs is not None:
                result_obj = result_obj.rio.write_crs(ref_crs, inplace=False)
        else:
            stack = np.stack(band_stacks[mode_key], axis=0)
            obs_counts = np.sum(np.isfinite(stack), axis=0)
            med = np.nanmedian(stack, axis=0).astype(np.float32)
            del stack
            band_stacks[mode_key].clear()

            valid_mask = np.isfinite(med)
            valid_pixels = int(np.count_nonzero(valid_mask))
            total_pixels = int(valid_mask.size)
            valid_pct = round(100.0 * valid_pixels / max(total_pixels, 1), 2)
            valid_area_ha = round(valid_pixels * px_area_ha, 3)
            med_obs = float(np.median(obs_counts[valid_mask])) if valid_pixels > 0 else 0.0

            spec = SPECTRAL_INDICES[mode_key]
            result_obj = xr.DataArray(
                med,
                dims=ref_da.dims,
                coords=ref_da.coords,
                name=f"{mode_key}_MEDIAN",
                attrs={
                    "sensor": sensor,
                    "index_name": mode_key,
                    "formula": spec["formula"],
                    "units": "dimensionless",
                    "long_name": f"Cloud-Free Median {mode_key} ({date_start} to {date_end})",
                    "datetime": f"{date_start}..{date_end}",
                    "date_start": date_start,
                    "date_end": date_end,
                    "scenes_used": len(dates_used),
                    "resolution_m": float(resolution),
                },
            )
            if ref_crs is not None:
                result_obj = result_obj.rio.write_crs(ref_crs, inplace=False)
            result_obj = result_obj.rio.write_nodata(np.nan, inplace=False)

    meta: dict[str, Any] = {
        "mode": mode_key,
        "sensor": sensor,
        "scenes_requested": total,
        "scenes_used": len(dates_used),
        "dates_used": dates_used,
        "skipped_dates": skipped_dates,
        "date_start": date_start,
        "date_end": date_end,
        "valid_pct": valid_pct,
        "valid_area_ha": valid_area_ha,
        "median_obs_per_px": round(med_obs, 1),
        "resolution_m": float(resolution),
        "crs": str(ref_crs) if ref_crs is not None else str(target_epsg),
    }

    del ref_da, band_stacks
    gc.collect()
    return result_obj, meta

