"""Memory-safe multi-sensor time-series extraction and pixel-aligned change detection.

Supports Sentinel-2 L2A, Landsat C2 L2 (1984–present), and Sentinel-1 SAR RTC,
processing strictly one epoch in RAM at a time (Streamlit Cloud 1 GB RAM safe)
and supporting exact polygon clipping (`rioxarray.rio.clip`).
"""

from __future__ import annotations

import gc
import io
import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np
import pandas as pd
import pystac
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling

from analize import (
    SAR_METRICS,
    SPECTRAL_INDICES,
    apply_cloud_mask,
    compute_index,
    compute_index_stats,
    extract_sar_metric,
    get_required_bands_for_sensor,
)
from geodezija_crs import clip_to_aoi_polygon
from stac_loader import SENSOR_LANDSAT, SENSOR_S1, SENSOR_S2, StacItemOrGroup, load_scene

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


def extract_time_series(
    items: Sequence[StacItemOrGroup],
    bbox_4326: Sequence[float],
    index_name: str,
    target_epsg: int | str = "EPSG:3794",
    resolution: float = 10,
    mask_clouds: bool = True,
    min_valid_pct: float = 10.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
    sensor: str = SENSOR_S2,
    aoi_polygon_4326: dict[str, Any] | None = None,
    min_swath_coverage_pct: float = 85.0,
) -> pd.DataFrame:
    """Extract a multi-temporal index or SAR backscatter time series (1 mosaicked date in RAM).

    Streams only the minimal bands needed for ``sensor`` and ``index_name``,
    mosaics all same-day tiles seamlessly onto the target CRS grid, clips to
    ``aoi_polygon_4326`` if provided, skips partial-swath dates with less than
    ``min_swath_coverage_pct`` non-NoData pixels before cloud masking, applies
    sensor-appropriate cloud masking (Sentinel-2 SCL or Landsat QA_PIXEL;
    skipped for all-weather SAR), records statistics, and immediately releases
    each raster via ``gc.collect()``.
    """
    idx_key = index_name.strip().upper()
    use_cloud_mask = mask_clouds and (sensor != SENSOR_S1)
    req_bands = get_required_bands_for_sensor(sensor, idx_key, include_cloud_mask=use_cloud_mask)

    epoch_groups = _normalize_epoch_groups(items)
    total = len(epoch_groups)
    rows: list[dict[str, Any]] = []
    skipped_epochs: list[dict[str, Any]] = []

    for step, day_items in enumerate(epoch_groups):
        dt_utc, date_str, scene_id, platform, cloud_cover = _group_metadata(day_items, sensor=sensor)

        if progress_callback is not None:
            progress_callback(step, total, date_str)

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
                logger.info(
                    "Skipped %s (%s): partial swath coverage %.1f%% < %.0f%%.",
                    date_str,
                    scene_id,
                    swath_pct,
                    min_swath_coverage_pct,
                )
                skipped_epochs.append(
                    {
                        "date_str": date_str,
                        "scene_id": str(ds.attrs.get("scene_id", scene_id)),
                        "valid_pct": swath_pct,
                        "reason": f"partial_swath ({swath_pct:.1f}% < {min_swath_coverage_pct:.0f}%)",
                    }
                )
                continue

            masked_ds, clear_pct = apply_cloud_mask(ds, mask_clouds=use_cloud_mask)
            layer_da = _compute_layer(masked_ds, sensor=sensor, metric_name=idx_key)
            stats = compute_index_stats(layer_da, resolution_m=float(resolution))

            valid_pct = float(stats["valid_pct"])
            if valid_pct < float(min_valid_pct):
                logger.info(
                    "Skipped %s (%s): %.1f%% valid pixels (clear: %.1f%%, min: %.0f%%).",
                    date_str,
                    scene_id,
                    valid_pct,
                    clear_pct,
                    min_valid_pct,
                )
                skipped_epochs.append(
                    {
                        "date_str": date_str,
                        "scene_id": scene_id,
                        "valid_pct": valid_pct,
                        "reason": "low_valid_pixels",
                    }
                )
            else:
                rows.append(
                    {
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
                    }
                )
        except Exception as exc:
            logger.warning("Failed to process epoch %s (%s): %s", date_str, scene_id, exc)
            skipped_epochs.append(
                {
                    "date_str": date_str,
                    "scene_id": scene_id,
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


def _stamp_banner_on_pil(img: Image.Image, stamp_text: str) -> Image.Image:
    """Draw a dark-slate (#0F172A) date & sensor stamp banner in the top-left corner."""
    if not stamp_text:
        return img

    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_stamp_font(size=max(13, min(18, rgba.width // 34)))

    bbox = draw.textbbox((0, 0), stamp_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x, pad_y = 10, 6
    margin = 10

    rect = [
        margin,
        margin,
        min(rgba.width - margin, margin + text_w + 2 * pad_x),
        min(rgba.height - margin, margin + text_h + 2 * pad_y),
    ]
    # Dark slate #0F172A background with crisp cartographic teal left accent #0F766E
    draw.rectangle(rect, fill=(15, 23, 42, 224))
    draw.rectangle([rect[0], rect[1], rect[0] + 4, rect[3]], fill=(15, 118, 110, 255))
    draw.text(
        (rect[0] + pad_x + 2, rect[1] + pad_y - bbox[1]),
        stamp_text,
        fill=(248, 250, 252, 255),
        font=font,
    )
    combined = Image.alpha_composite(rgba, overlay).convert("RGB")
    return combined


def _stretch_rgb_for_pil(
    ds: xr.Dataset,
    bands: Sequence[str],
    y_coords: np.ndarray,
) -> np.ndarray:
    """Compose a north-up uint8 (H, W, 3) array using 2–98% reflectance stretch."""
    channels: list[np.ndarray] = []
    valid = np.ones(ds[bands[0]].shape, dtype=bool)
    for b in bands:
        arr = ds[b].values.astype(np.float32, copy=False)
        valid &= np.isfinite(arr)
        channels.append(arr)

    bg_rgb = (226, 232, 240)  # Neutral cartographic slate #E2E8F0
    out_ch: list[np.ndarray] = []
    for idx, arr in enumerate(channels):
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            stretched = np.zeros_like(arr, dtype=np.float32)
        else:
            lo, hi = np.percentile(fin, [2.0, 98.0])
            if hi <= lo:
                lo, hi = float(fin.min()), float(fin.max())
            denom = max(float(hi - lo), 1e-6)
            stretched = np.clip((arr - np.float32(lo)) / np.float32(denom), 0.0, 1.0)
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
    img = _resize_pil_frame(img, max_width=max_width, target_size=target_size)
    if stamp_text:
        img = _stamp_banner_on_pil(img, stamp_text)
    return img


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Encode a PIL Image to compact PNG bytes in memory."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def compile_timelapse_gif(
    frames_png_bytes: Sequence[bytes],
    duration_ms: int = 800,
) -> bytes:
    """Compile a sequence of PNG frame byte buffers into a looping animated GIF."""
    if not frames_png_bytes:
        raise ValueError("Cannot compile timelapse GIF from an empty frame list.")

    pil_frames: list[Image.Image] = []
    base_size: tuple[int, int] | None = None
    for raw in frames_png_bytes:
        with Image.open(io.BytesIO(raw)) as im:
            frame_rgb = im.convert("RGB")
            if base_size is None:
                base_size = frame_rgb.size
            elif frame_rgb.size != base_size:
                frame_rgb = frame_rgb.resize(base_size, resample=Image.Resampling.BILINEAR)
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
    del pil_frames
    gc.collect()
    return gif_bytes


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
    min_valid_pct: float = 10.0,
    duration_ms: int = 800,
    max_width: int = 700,
    progress_callback: Callable[[int, int, str], None] | None = None,
    min_swath_coverage_pct: float = 85.0,
) -> dict[str, Any]:
    """Generate a memory-safe visual timelapse (compact PNG frames + animated GIF).

    Loads strictly one mosaicked acquisition date at a time, skips partial-swath
    frames (< ``min_swath_coverage_pct`` non-NoData pixels before cloud masking),
    aligns frames onto a consistent pixel grid, stamps each frame with
    ``YYYY-MM-DD | Platform``, frees raster memory via ``gc.collect()`` after
    every frame, and compiles a looping GIF.
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

    epoch_groups = _normalize_epoch_groups(items)
    total = len(epoch_groups)

    frames: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    target_pil_size: tuple[int, int] | None = None

    for step, day_items in enumerate(epoch_groups):
        _, date_str, scene_id, platform_raw, _ = _group_metadata(day_items, sensor=sensor)
        if progress_callback is not None:
            progress_callback(step, total, date_str)

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
                logger.info(
                    "Timelapse skipped %s (%s): partial swath coverage %.1f%% < %.0f%%.",
                    date_str,
                    merged_scene_id,
                    swath_pct,
                    min_swath_coverage_pct,
                )
                skipped.append(
                    {
                        "date_str": date_str,
                        "scene_id": merged_scene_id,
                        "valid_pct": swath_pct,
                        "reason": f"partial_swath ({swath_pct:.1f}% < {min_swath_coverage_pct:.0f}%)",
                    }
                )
                continue

            if is_rgb_composite:
                # Check valid pixel coverage across the 3 composite bands
                valid = np.ones(ds[req_bands[0]].shape, dtype=bool)
                for b in req_bands:
                    valid &= np.isfinite(ds[b].values)
                valid_pct = float(100.0 * np.count_nonzero(valid) / max(valid.size, 1))
                if valid_pct < float(min_valid_pct):
                    skipped.append(
                        {"date_str": date_str, "scene_id": merged_scene_id, "valid_pct": round(valid_pct, 1), "reason": "low_valid_pixels"}
                    )
                    continue

                y_coords = np.asarray(ds.coords["y"].values, dtype=np.float64)
                rgb_u8 = _stretch_rgb_for_pil(ds, req_bands, y_coords)
                img = Image.fromarray(rgb_u8, mode="RGB")
                img = _resize_pil_frame(img, max_width=max_width, target_size=target_pil_size)
                if target_pil_size is None:
                    target_pil_size = img.size
                img = _stamp_banner_on_pil(img, stamp)
            else:
                masked_ds, _ = apply_cloud_mask(ds, mask_clouds=use_cloud_mask)
                layer_da = _compute_layer(masked_ds, sensor=sensor, metric_name=mode)
                stats = compute_index_stats(layer_da, resolution_m=float(resolution))
                valid_pct = float(stats["valid_pct"])
                if valid_pct < float(min_valid_pct):
                    skipped.append(
                        {"date_str": date_str, "scene_id": merged_scene_id, "valid_pct": round(valid_pct, 1), "reason": "low_valid_pixels"}
                    )
                    continue

                spec = SAR_METRICS[mode] if is_sar else SPECTRAL_INDICES[mode]
                img = render_dataarray_to_pil(
                    layer_da,
                    cmap_name=str(spec["cmap"]),
                    vmin=float(spec["vmin"]),
                    vmax=float(spec["vmax"]),
                    stamp_text=stamp,
                    max_width=max_width,
                    target_size=target_pil_size,
                )
                if target_pil_size is None:
                    target_pil_size = img.size

            frames.append(
                {
                    "date_str": date_str,
                    "platform": platform_lbl,
                    "scene_id": merged_scene_id,
                    "valid_pct": round(valid_pct, 1),
                    "png_bytes": pil_to_png_bytes(img),
                }
            )
        except Exception as exc:
            logger.warning("Timelapse skipped %s (%s): %s", date_str, scene_id, exc)
            skipped.append(
                {"date_str": date_str, "scene_id": scene_id, "valid_pct": 0.0, "reason": str(exc)}
            )
        finally:
            del ds, masked_ds, layer_da, img
            gc.collect()

    if progress_callback is not None:
        progress_callback(total, total, "done")

    gif_bytes = (
        compile_timelapse_gif([f["png_bytes"] for f in frames], duration_ms=duration_ms)
        if frames
        else b""
    )

    return {
        "frames": frames,
        "gif_bytes": gif_bytes,
        "skipped": skipped,
        "layer_mode": mode,
        "sensor": sensor,
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

    Streams strictly one same-day mosaicked acquisition at a time (1 GB RAM safe),
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

    for step, day_items in enumerate(selected_groups):
        _, date_str, scene_id, _, _ = _group_metadata(day_items, sensor=sensor)
        if progress_callback is not None:
            progress_callback(step, total, date_str)

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
            masked_ds, clear_pct = apply_cloud_mask(ds, mask_clouds=True)
            if clear_pct < float(min_valid_pct):
                skipped_dates.append(
                    {"date_str": date_str, "scene_id": scene_id, "valid_pct": round(clear_pct, 1), "reason": "cloudy"}
                )
                continue

            if is_rgb_nir:
                if ref_da is None:
                    ref_da = masked_ds[COMPOSITE_RGB_NIR_BANDS[0]].copy()
                    ref_crs = masked_ds.rio.crs or ref_da.rio.crs
                for b in COMPOSITE_RGB_NIR_BANDS:
                    da_b = masked_ds[b]
                    if not _grids_match(ref_da, da_b):
                        da_b = da_b.rio.reproject_match(ref_da, resampling=Resampling.bilinear, nodata=np.nan)
                    band_stacks[b].append(da_b.values.astype(np.float32, copy=True))
            else:
                layer_da = compute_index(masked_ds, mode_key)
                if ref_da is None:
                    ref_da = layer_da.copy()
                    ref_crs = layer_da.rio.crs or masked_ds.rio.crs
                elif not _grids_match(ref_da, layer_da):
                    layer_da = layer_da.rio.reproject_match(ref_da, resampling=Resampling.bilinear, nodata=np.nan)
                band_stacks[mode_key].append(layer_da.values.astype(np.float32, copy=True))

            dates_used.append(date_str)
        except Exception as exc:
            logger.warning("Composite skipped %s (%s): %s", date_str, scene_id, exc)
            skipped_dates.append(
                {"date_str": date_str, "scene_id": scene_id, "valid_pct": 0.0, "reason": str(exc)}
            )
        finally:
            del ds, masked_ds, layer_da
            gc.collect()

    if progress_callback is not None:
        progress_callback(total, total, "done")

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

