"""Rapid Natural Disaster Assessment Module (Before / During / After).

Supports 3 core European/Slovenian geodetic remote-sensing disaster workflows
strictly in projected metric coordinates (D96/TM ``EPSG:3794`` or UTM 33N ``EPSG:32633``):

1. **Flood Detection (Sentinel-1 SAR RTC):** All-weather dual-condition backscatter
   change separating permanent water (rivers/lakes) from newly flooded land:
   ``(VV_event < threshold_db) & (VV_pre >= threshold_db) & (VV_event - VV_pre < -2.5 dB)``.
2. **Wildfire Burn Severity (Sentinel-2 L2A / Landsat C2 L2 dNBR):** Cloud-masked
   ``dNBR = NBR_pre - NBR_post`` classified into official USGS Fire Severity Classes.
3. **Forest & Vegetation Disturbance (Sentinel-2 L2A / Landsat C2 L2 dNDVI):**
   Cloud-masked ``ΔNDVI = NDVI_post - NDVI_pre`` identifying severe vegetation loss
   (windthrow / clearcut: ``ΔNDVI < -0.25``) and moderate loss / drought
   (``-0.25 <= ΔNDVI < -0.12``).

Includes built-in real Slovenian case studies (2023 Savinjska/Celje floods,
2022 Kras wildfire, 2023 Ljubljansko barje floods) for rapid 3-click analysis.
"""

from __future__ import annotations

import gc
import io
from datetime import date, datetime, timedelta
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.enums import Resampling
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr

from analize import apply_cloud_mask, compute_index
from geodezija_crs import CRS_D96TM, export_to_geotiff_bytes
from stac_loader import (
    DEFAULT_MIN_AOI_COVERAGE_PCT,
    INTERVAL_ALL,
    SENSOR_LANDSAT,
    SENSOR_S1,
    SENSOR_S2,
    load_scene,
    search_scenes,
)

# ---------------------------------------------------------------------------
# Disaster Type Identifiers & Built-in Slovenian Case Studies
# ---------------------------------------------------------------------------
DISASTER_FLOOD: str = "flood"
DISASTER_WILDFIRE: str = "wildfire"
DISASTER_FOREST: str = "forest_loss"

DISASTER_TYPES: tuple[str, ...] = (
    DISASTER_FLOOD,
    DISASTER_WILDFIRE,
    DISASTER_FOREST,
)

SLOVENIA_CASE_STUDIES: dict[str, dict[str, Any]] = {
    "custom": {
        "id": "custom",
        "name_sl": "Po meri (trenutni AOI)",
        "name_en": "Custom (current AOI)",
        "desc_sl": "Uporabi trenutno izbrano območje AOI in datum dogodka.",
        "desc_en": "Uses your currently selected AOI and event date.",
        "lat": None,
        "lon": None,
        "radius_m": None,
        "event_date": None,
        "disaster_type": None,
    },
    "celje_flood_2023": {
        "id": "celje_flood_2023",
        "name_sl": "Celje Poplave 2023",
        "name_en": "Celje Floods 2023",
        "full_title_sl": "Poplave — Savinjska dolina / Celje (avgust 2023)",
        "full_title_en": "Floods — Savinja Valley / Celje (August 2023)",
        "desc_sl": "Katastrofalne poplave reke Savinje (4. avgust 2023) — Sentinel-1 SAR RTC.",
        "desc_en": "Catastrophic Savinja River floods (August 4, 2023) — Sentinel-1 SAR RTC.",
        "lat": 46.2397,
        "lon": 15.2677,
        "radius_m": 6000,
        "event_date": date(2023, 8, 4),
        "disaster_type": DISASTER_FLOOD,
    },
    "kras_fire_2022": {
        "id": "kras_fire_2022",
        "name_sl": "Kras Požar 2022",
        "name_en": "Kras Wildfire 2022",
        "full_title_sl": "Požar na Krasu — Kostanjevica / Cerje (julij 2022)",
        "full_title_en": "Kras Wildfire — Kostanjevica / Cerje (July 2022)",
        "desc_sl": "Največji požar v zgodovini Slovenije (julij 2022) — Sentinel-2 dNBR (USGS).",
        "desc_en": "Largest wildfire in Slovenian history (July 2022) — Sentinel-2 dNBR (USGS).",
        "lat": 45.8480,
        "lon": 13.6350,
        "radius_m": 6000,
        "event_date": date(2022, 7, 22),
        "disaster_type": DISASTER_WILDFIRE,
    },
    "barje_flood_2023": {
        "id": "barje_flood_2023",
        "name_sl": "Barje Poplave 2023",
        "name_en": "Barje Floods 2023",
        "full_title_sl": "Poplave — Ljubljansko barje (avgust 2023)",
        "full_title_en": "Floods — Ljubljana Marshes (August 2023)",
        "desc_sl": "Razlitje Ljubljanice in pritokov (5. avgust 2023) — Sentinel-1 SAR RTC.",
        "desc_en": "Ljubljanica & tributaries flooding (August 5, 2023) — Sentinel-1 SAR RTC.",
        "lat": 46.0050,
        "lon": 14.4600,
        "radius_m": 5000,
        "event_date": date(2023, 8, 5),
        "disaster_type": DISASTER_FLOOD,
    },
}

# ---------------------------------------------------------------------------
# Class Palettes & Specifications (No purple/violet — crisp cartographic colors)
# ---------------------------------------------------------------------------
FLOOD_CLASS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "code": 0,
        "sl": "Nepoplavljeno kopno",
        "en": "Unflooded Land",
        "hex": "#CBD5E1",
        "rgba_slippy": (203, 213, 225, 0),  # transparent on slippy map
        "affected": False,
    },
    {
        "code": 1,
        "sl": "Stalna voda (reke, jezera)",
        "en": "Permanent Water (rivers, lakes)",
        "hex": "#1D4ED8",
        "rgba_slippy": (29, 78, 216, 205),
        "affected": False,
    },
    {
        "code": 2,
        "sl": "Novo poplavljeno območje",
        "en": "Newly Flooded Area",
        "hex": "#06B6D4",
        "rgba_slippy": (6, 182, 212, 240),
        "affected": True,
    },
)

WILDFIRE_CLASS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "code": 0,
        "sl": "Nepoškodovano / Regeneracija (dNBR < 0,10)",
        "en": "Unburned / Regrowth (dNBR < 0.10)",
        "hex": "#15803D",
        "rgba_slippy": (21, 128, 61, 0),  # keep unburned transparent on slippy map
        "affected": False,
    },
    {
        "code": 1,
        "sl": "Nizka stopnja poškodovanosti (0,10–0,27)",
        "en": "Low Severity (0.10–0.27)",
        "hex": "#FACC15",
        "rgba_slippy": (250, 204, 21, 225),
        "affected": True,
    },
    {
        "code": 2,
        "sl": "Zmerno nizka poškodovanost (0,27–0,44)",
        "en": "Moderate-Low Severity (0.27–0.44)",
        "hex": "#FB923C",
        "rgba_slippy": (251, 146, 60, 235),
        "affected": True,
    },
    {
        "code": 3,
        "sl": "Zmerno visoka poškodovanost (0,44–0,66)",
        "en": "Moderate-High Severity (0.44–0.66)",
        "hex": "#EA580C",
        "rgba_slippy": (234, 88, 12, 242),
        "affected": True,
    },
    {
        "code": 4,
        "sl": "Visoka stopnja poškodovanosti (dNBR ≥ 0,66)",
        "en": "High Severity (dNBR ≥ 0.66)",
        "hex": "#B91C1C",
        "rgba_slippy": (185, 28, 28, 248),
        "affected": True,
    },
)

FOREST_CLASS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "code": 0,
        "sl": "Stabilna vegetacija (ΔNDVI ≥ −0,12)",
        "en": "Stable Vegetation (ΔNDVI ≥ −0.12)",
        "hex": "#CBD5E1",
        "rgba_slippy": (203, 213, 225, 0),
        "affected": False,
    },
    {
        "code": 1,
        "sl": "Zmerna izguba vegetacije / Suša (−0,25 do −0,12)",
        "en": "Moderate Vegetation Loss / Drought (−0.25 to −0.12)",
        "hex": "#F59E0B",
        "rgba_slippy": (245, 158, 11, 225),
        "affected": True,
    },
    {
        "code": 2,
        "sl": "Močna izguba vegetacije / Vetrolom / Posek (ΔNDVI < −0,25)",
        "en": "Severe Vegetation Loss / Windthrow / Clearcut (ΔNDVI < −0.25)",
        "hex": "#DC2626",
        "rgba_slippy": (220, 38, 38, 242),
        "affected": True,
    },
)

NODATA_U8: int = 255


def get_class_specs(disaster_type: str) -> tuple[dict[str, Any], ...]:
    """Return the discrete class specification tuple for a disaster type."""
    if disaster_type == DISASTER_FLOOD:
        return FLOOD_CLASS_SPECS
    if disaster_type == DISASTER_WILDFIRE:
        return WILDFIRE_CLASS_SPECS
    if disaster_type == DISASTER_FOREST:
        return FOREST_CLASS_SPECS
    raise ValueError(f"Unsupported disaster_type: {disaster_type!r}")


def build_slippy_rgba_map(disaster_type: str) -> dict[int, tuple[int, int, int, int]]:
    """Build the ``discrete_rgba_map`` dictionary for ``map_overlay.raster_to_map_overlay_png``."""
    specs = get_class_specs(disaster_type)
    mapping: dict[int, tuple[int, int, int, int]] = {NODATA_U8: (0, 0, 0, 0)}
    for item in specs:
        mapping[int(item["code"])] = tuple(item["rgba_slippy"])
    return mapping


def build_carto_discrete_items(
    disaster_type: str,
    lang: str = "sl",
) -> list[tuple[int, str, str]]:
    """Build ``(code, label, hex)`` tuples for ``kartografija.render_geodetic_map``."""
    specs = get_class_specs(disaster_type)
    lbl_key = "en" if lang == "en" else "sl"
    # Order impacted classes first for clear thesis cartographic legends
    ordered = [s for s in specs if s["affected"]] + [s for s in specs if not s["affected"]]
    return [(int(s["code"]), str(s[lbl_key]), str(s["hex"])) for s in ordered]


# ---------------------------------------------------------------------------
# Geodetic Grid Alignment & Metric Area Helpers
# ---------------------------------------------------------------------------
def _pixel_area_ha(da: xr.DataArray, fallback_resolution_m: float = 10.0) -> float:
    """Compute exact metric pixel area in hectares from projected x/y coordinates."""
    try:
        x_vals = np.asarray(da.coords["x"].values, dtype=np.float64)
        y_vals = np.asarray(da.coords["y"].values, dtype=np.float64)
        if x_vals.size > 1 and y_vals.size > 1:
            dx = float(abs(np.median(np.diff(x_vals))))
            dy = float(abs(np.median(np.diff(y_vals))))
            if dx > 0.1 and dy > 0.1:
                return (dx * dy) / 10_000.0
    except Exception:
        pass
    res = max(float(fallback_resolution_m), 1.0)
    return (res * res) / 10_000.0


def align_rasters_exact(da_ref: xr.DataArray, da_other: xr.DataArray) -> xr.DataArray:
    """Align ``da_other`` onto the exact projected pixel grid of ``da_ref``."""
    same_shape = da_ref.shape == da_other.shape
    same_coords = False
    if same_shape:
        try:
            same_coords = bool(
                np.allclose(da_ref.coords["x"].values, da_other.coords["x"].values, atol=1e-2)
                and np.allclose(da_ref.coords["y"].values, da_other.coords["y"].values, atol=1e-2)
            )
        except Exception:
            same_coords = False
    if same_coords:
        return da_other
    return da_other.rio.reproject_match(da_ref, resampling=Resampling.bilinear)


def _summarize_classified_raster(
    class_da: xr.DataArray,
    disaster_type: str,
    resolution_m: float = 10.0,
    lang: str = "sl",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute exact metric areas [ha & km²] and percentages for every class."""
    specs = get_class_specs(disaster_type)
    lbl_key = "en" if lang == "en" else "sl"
    px_ha = _pixel_area_ha(class_da, fallback_resolution_m=resolution_m)

    vals = np.asarray(class_da.values, dtype=np.uint8)
    valid_mask = vals != np.uint8(NODATA_U8)
    valid_pixels = int(np.count_nonzero(valid_mask))
    valid_area_ha = valid_pixels * px_ha

    rows: list[dict[str, Any]] = []
    affected_pixels = 0
    for item in specs:
        code = int(item["code"])
        cnt = int(np.count_nonzero(vals == np.uint8(code)))
        area_ha = cnt * px_ha
        pct = (100.0 * cnt / valid_pixels) if valid_pixels > 0 else 0.0
        if item["affected"]:
            affected_pixels += cnt
        rows.append(
            {
                "code": code,
                "class_label": str(item[lbl_key]),
                "hex": str(item["hex"]),
                "affected": bool(item["affected"]),
                "pixels": cnt,
                "area_ha": round(area_ha, 2),
                "area_km2": round(area_ha / 100.0, 4),
                "pct": round(pct, 2),
            }
        )

    affected_ha = affected_pixels * px_ha
    affected_pct = (100.0 * affected_pixels / valid_pixels) if valid_pixels > 0 else 0.0

    summary: dict[str, Any] = {
        "valid_pixels": valid_pixels,
        "valid_area_ha": round(valid_area_ha, 2),
        "valid_area_km2": round(valid_area_ha / 100.0, 4),
        "affected_pixels": affected_pixels,
        "affected_area_ha": round(affected_ha, 2),
        "affected_area_km2": round(affected_ha / 100.0, 4),
        "affected_pct": round(affected_pct, 2),
        "pixel_area_ha": px_ha,
    }
    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# 1. Core Classification Functions (Pure Geodetic Logic — Unit Testable)
# ---------------------------------------------------------------------------
def classify_flood_sar(
    vv_pre: xr.DataArray,
    vv_event: xr.DataArray,
    *,
    threshold_db: float = -16.0,
    delta_db: float = -2.5,
    resolution_m: float = 10.0,
    lang: str = "sl",
) -> tuple[xr.DataArray, pd.DataFrame, dict[str, Any]]:
    """Classify Sentinel-1 SAR flood impact separating permanent vs. newly flooded water.

    Dual-condition SAR flood detection:
    - Newly flooded (``code = 2``):
      ``(VV_event < threshold_db) & (VV_pre >= threshold_db) & ((VV_event - VV_pre) < delta_db)``
    - Permanent water (``code = 1``):
      ``(VV_pre < threshold_db) & (VV_event < threshold_db)``
    - Unflooded land (``code = 0``): remaining valid pixels
    - NoData (``code = 255``): NaN in either pre or event raster
    """
    vv_ev_aligned = align_rasters_exact(vv_pre, vv_event)
    pre_arr = np.asarray(vv_pre.values, dtype=np.float32)
    ev_arr = np.asarray(vv_ev_aligned.values, dtype=np.float32)

    valid = np.isfinite(pre_arr) & np.isfinite(ev_arr)
    diff = ev_arr - pre_arr

    newly_flooded = (
        valid
        & (ev_arr < float(threshold_db))
        & (pre_arr >= float(threshold_db))
        & (diff < float(delta_db))
    )
    permanent_water = valid & (pre_arr < float(threshold_db)) & (~newly_flooded)

    cls = np.full(pre_arr.shape, NODATA_U8, dtype=np.uint8)
    cls[valid] = 0
    cls[permanent_water] = 1
    cls[newly_flooded] = 2

    class_da = xr.DataArray(
        cls,
        dims=vv_pre.dims,
        coords=vv_pre.coords,
        name="flood_class",
    )
    if vv_pre.rio.crs is not None:
        class_da = class_da.rio.write_crs(vv_pre.rio.crs, inplace=False)
    class_da = class_da.rio.write_nodata(NODATA_U8, inplace=False)

    df, summary = _summarize_classified_raster(
        class_da, DISASTER_FLOOD, resolution_m=resolution_m, lang=lang
    )
    perm_row = df.loc[df["code"] == 1]
    summary["permanent_water_ha"] = float(perm_row["area_ha"].iloc[0]) if not perm_row.empty else 0.0
    summary["newly_flooded_ha"] = float(summary["affected_area_ha"])
    summary["threshold_db"] = float(threshold_db)
    summary["delta_db"] = float(delta_db)
    return class_da, df, summary


def classify_wildfire_dnbr(
    nbr_pre: xr.DataArray,
    nbr_post: xr.DataArray,
    *,
    resolution_m: float = 10.0,
    lang: str = "sl",
) -> tuple[xr.DataArray, xr.DataArray, pd.DataFrame, dict[str, Any]]:
    """Compute ``dNBR = NBR_pre - NBR_post`` and classify into official USGS Fire Severity Classes.

    USGS Fire Severity Thresholds:
    - ``0``: Unburned / Regrowth (``dNBR < 0.10``)
    - ``1``: Low Severity (``0.10 <= dNBR < 0.27``)
    - ``2``: Moderate-Low Severity (``0.27 <= dNBR < 0.44``)
    - ``3``: Moderate-High Severity (``0.44 <= dNBR < 0.66``)
    - ``4``: High Severity (``dNBR >= 0.66``)
    """
    nbr_post_aligned = align_rasters_exact(nbr_pre, nbr_post)
    pre_arr = np.asarray(nbr_pre.values, dtype=np.float32)
    post_arr = np.asarray(nbr_post_aligned.values, dtype=np.float32)

    valid = np.isfinite(pre_arr) & np.isfinite(post_arr)
    dnbr_arr = np.where(valid, pre_arr - post_arr, np.nan).astype(np.float32)

    cls = np.full(pre_arr.shape, NODATA_U8, dtype=np.uint8)
    cls[valid & (dnbr_arr < 0.10)] = 0
    cls[valid & (dnbr_arr >= 0.10) & (dnbr_arr < 0.27)] = 1
    cls[valid & (dnbr_arr >= 0.27) & (dnbr_arr < 0.44)] = 2
    cls[valid & (dnbr_arr >= 0.44) & (dnbr_arr < 0.66)] = 3
    cls[valid & (dnbr_arr >= 0.66)] = 4

    class_da = xr.DataArray(
        cls,
        dims=nbr_pre.dims,
        coords=nbr_pre.coords,
        name="burn_severity_class",
    )
    dnbr_da = xr.DataArray(
        dnbr_arr,
        dims=nbr_pre.dims,
        coords=nbr_pre.coords,
        name="dNBR",
    )
    if nbr_pre.rio.crs is not None:
        class_da = class_da.rio.write_crs(nbr_pre.rio.crs, inplace=False)
        dnbr_da = dnbr_da.rio.write_crs(nbr_pre.rio.crs, inplace=False)
    class_da = class_da.rio.write_nodata(NODATA_U8, inplace=False)

    df, summary = _summarize_classified_raster(
        class_da, DISASTER_WILDFIRE, resolution_m=resolution_m, lang=lang
    )
    high_sev_ha = float(df.loc[df["code"].isin([3, 4]), "area_ha"].sum())
    summary["high_severity_ha"] = round(high_sev_ha, 2)
    valid_dnbr = dnbr_arr[valid]
    summary["mean_dnbr"] = round(float(np.mean(valid_dnbr)), 3) if valid_dnbr.size else 0.0
    return class_da, dnbr_da, df, summary


def classify_forest_dndvi(
    ndvi_pre: xr.DataArray,
    ndvi_post: xr.DataArray,
    *,
    severe_threshold: float = -0.25,
    moderate_threshold: float = -0.12,
    resolution_m: float = 10.0,
    lang: str = "sl",
) -> tuple[xr.DataArray, xr.DataArray, pd.DataFrame, dict[str, Any]]:
    """Compute ``ΔNDVI = NDVI_post - NDVI_pre`` and classify forest / vegetation disturbance.

    - ``2``: Severe vegetation loss / windthrow / clearcut (``ΔNDVI < -0.25``)
    - ``1``: Moderate vegetation loss / drought (``-0.25 <= ΔNDVI < -0.12``)
    - ``0``: Stable vegetation (``ΔNDVI >= -0.12``)
    """
    ndvi_post_aligned = align_rasters_exact(ndvi_pre, ndvi_post)
    pre_arr = np.asarray(ndvi_pre.values, dtype=np.float32)
    post_arr = np.asarray(ndvi_post_aligned.values, dtype=np.float32)

    valid = np.isfinite(pre_arr) & np.isfinite(post_arr)
    dndvi_arr = np.where(valid, post_arr - pre_arr, np.nan).astype(np.float32)

    cls = np.full(pre_arr.shape, NODATA_U8, dtype=np.uint8)
    cls[valid & (dndvi_arr >= float(moderate_threshold))] = 0
    cls[valid & (dndvi_arr < float(moderate_threshold)) & (dndvi_arr >= float(severe_threshold))] = 1
    cls[valid & (dndvi_arr < float(severe_threshold))] = 2

    class_da = xr.DataArray(
        cls,
        dims=ndvi_pre.dims,
        coords=ndvi_pre.coords,
        name="forest_loss_class",
    )
    dndvi_da = xr.DataArray(
        dndvi_arr,
        dims=ndvi_pre.dims,
        coords=ndvi_pre.coords,
        name="dNDVI",
    )
    if ndvi_pre.rio.crs is not None:
        class_da = class_da.rio.write_crs(ndvi_pre.rio.crs, inplace=False)
        dndvi_da = dndvi_da.rio.write_crs(ndvi_pre.rio.crs, inplace=False)
    class_da = class_da.rio.write_nodata(NODATA_U8, inplace=False)

    df, summary = _summarize_classified_raster(
        class_da, DISASTER_FOREST, resolution_m=resolution_m, lang=lang
    )
    sev_row = df.loc[df["code"] == 2]
    mod_row = df.loc[df["code"] == 1]
    summary["severe_loss_ha"] = float(sev_row["area_ha"].iloc[0]) if not sev_row.empty else 0.0
    summary["moderate_loss_ha"] = float(mod_row["area_ha"].iloc[0]) if not mod_row.empty else 0.0
    valid_d = dndvi_arr[valid]
    summary["mean_dndvi"] = round(float(np.mean(valid_d)), 3) if valid_d.size else 0.0
    return class_da, dndvi_da, df, summary


# ---------------------------------------------------------------------------
# 2. Swipe Image Pair Rendering (RGB for Optical, VV dB for SAR)
# ---------------------------------------------------------------------------
def _north_up(arr: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    if y_coords.size > 1 and float(y_coords[0]) < float(y_coords[-1]):
        return np.flipud(arr)
    return arr


def _stretch_band_u8(arr: np.ndarray, lo_hi: tuple[float, float] | None = None) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.full(data.shape, 226, dtype=np.uint8)
    if lo_hi is None:
        lo = float(np.percentile(valid, 2.0))
        hi = float(np.percentile(valid, 98.0))
    else:
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)


def render_disaster_swipe_pair(
    ds_pre: xr.Dataset,
    ds_post: xr.Dataset,
    *,
    is_sar: bool = False,
    max_width: int = 760,
) -> tuple[bytes, bytes]:
    """Render identical-size Before/After PNG bytes (RGB for optical, VV dB for SAR)."""
    imgs: list[Image.Image] = []
    target_size: tuple[int, int] | None = None

    for ds in (ds_pre, ds_post):
        y_coords = np.asarray(ds.coords["y"].values, dtype=np.float64)
        if is_sar:
            vv = _north_up(np.asarray(ds["VV"].values, dtype=np.float32), y_coords)
            finite = np.isfinite(vv)
            norm = np.clip((np.nan_to_num(vv, nan=-22.0) - (-22.0)) / 20.0, 0.0, 1.0)
            cmap = matplotlib.colormaps["Greys_r"]
            rgb_u8 = np.clip(np.round(cmap(norm)[..., :3] * 255.0), 0, 255).astype(np.uint8)
            rgb_u8[~finite] = np.array([226, 232, 240], dtype=np.uint8)
        else:
            r = _north_up(np.asarray(ds["B04"].values, dtype=np.float32), y_coords)
            g = _north_up(np.asarray(ds["B03"].values, dtype=np.float32), y_coords)
            b = _north_up(np.asarray(ds["B02"].values, dtype=np.float32), y_coords)
            finite = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
            rgb_u8 = np.stack(
                [_stretch_band_u8(r), _stretch_band_u8(g), _stretch_band_u8(b)], axis=-1
            )
            rgb_u8[~finite] = np.array([226, 232, 240], dtype=np.uint8)

        im = Image.fromarray(rgb_u8, mode="RGB")
        if target_size is None:
            if im.width > max_width:
                scale = float(max_width) / float(im.width)
                im = im.resize(
                    (max_width, max(1, int(round(im.height * scale)))),
                    resample=Image.Resampling.BILINEAR,
                )
            target_size = im.size
        elif im.size != target_size:
            im = im.resize(target_size, resample=Image.Resampling.BILINEAR)
        imgs.append(im)

    out_bytes: list[bytes] = []
    for im in imgs:
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        out_bytes.append(buf.getvalue())
    return out_bytes[0], out_bytes[1]


# ---------------------------------------------------------------------------
# 3. Automated STAC Before / During / After Scene Selection & Pipeline
# ---------------------------------------------------------------------------
def _pick_best_scene(
    items: list[list[Any]],
    df: pd.DataFrame,
    target_date: date,
    *,
    is_sar: bool = False,
) -> tuple[list[Any], pd.Series] | None:
    """Pick the best scene group from search results (highest coverage, clearest, closest to target_date)."""
    if not items or df.empty:
        return None

    best_idx = 0
    best_score: tuple[float, float, float] | None = None
    for idx in range(len(df)):
        row = df.iloc[idx]
        try:
            d_obj = datetime.strptime(str(row["date_str"]), "%Y-%m-%d").date()
        except Exception:
            continue
        day_dist = float(abs((d_obj - target_date).days))
        cov = float(row.get("aoi_coverage_pct", 100.0)) if not pd.isna(row.get("aoi_coverage_pct", 100.0)) else 100.0
        # Prefer >=95% coverage first, then low cloud (optical), then closest date
        cov_bucket = 0.0 if cov >= 95.0 else (1.0 if cov >= 85.0 else 2.0)
        if is_sar:
            score = (cov_bucket, day_dist, -cov)
        else:
            cc = float(row["cloud_cover"]) if not pd.isna(row["cloud_cover"]) else 50.0
            score = (cov_bucket, cc * 1.5 + day_dist * 0.4, day_dist)
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    return items[best_idx], df.iloc[best_idx]


def _search_window_with_fallback(
    *,
    sensor: str,
    bbox_4326: Sequence[float],
    start_date: date,
    end_date: date,
    max_cloud: float,
    aoi_polygon_4326: dict[str, Any] | None,
) -> tuple[list[list[Any]], pd.DataFrame]:
    """Search a STAC date window strictly (>=90% AOI coverage) with graceful fallback."""
    today = date.today()
    s_d = min(start_date, today)
    e_d = min(end_date, today)
    if s_d > e_d:
        return [], pd.DataFrame()

    is_sar = sensor == SENSOR_S1
    # Pass 1: Strict >=90% AOI coverage & low cloud
    items, df = search_scenes(
        sensor=sensor,
        bbox_4326=bbox_4326,
        start_date=s_d,
        end_date=e_d,
        max_cloud_cover=max_cloud,
        exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=DEFAULT_MIN_AOI_COVERAGE_PCT,
        interval_mode=INTERVAL_ALL,
    )
    if items and not df.empty:
        return items, df

    # Pass 2: Slightly relaxed cloud (optical) or >=70% coverage (SAR / edge swaths)
    items, df = search_scenes(
        sensor=sensor,
        bbox_4326=bbox_4326,
        start_date=s_d,
        end_date=e_d,
        max_cloud_cover=min(max_cloud + 20.0, 50.0) if not is_sar else 100.0,
        exclude_l7_slc_off=(sensor == SENSOR_LANDSAT),
        aoi_polygon_4326=aoi_polygon_4326,
        min_coverage_pct=70.0,
        interval_mode=INTERVAL_ALL,
    )
    return items, df


def select_sensor_for_disaster(disaster_type: str, event_date: date) -> str:
    """Automatically choose the optimal satellite sensor for a disaster type and date."""
    if disaster_type == DISASTER_FLOOD:
        return SENSOR_S1
    if event_date < date(2016, 1, 1):
        return SENSOR_LANDSAT
    return SENSOR_S2


def run_disaster_assessment(
    *,
    disaster_type: str,
    bbox_4326: Sequence[float],
    event_date: date,
    target_crs: str = CRS_D96TM,
    resolution: int = 20,
    search_window_days: int = 60,
    flood_threshold_db: float = -16.0,
    flood_delta_db: float = -2.5,
    aoi_polygon_4326: dict[str, Any] | None = None,
    lang: str = "sl",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run an end-to-end Before / During / After natural disaster assessment in metric CRS.

    Returns a dictionary containing:
    - ``disaster_type``, ``sensor``, ``crs``, ``resolution_m``
    - ``date_pre``, ``date_event``, ``date_post`` (optional recovery/during dates)
    - ``class_da`` (uint8 metric DataArray), ``continuous_da`` (dNBR/dNDVI or None)
    - ``breakdown_df`` (exact ha / km² / % per class), ``summary`` (KPI dict)
    - ``swipe_png_pre``, ``swipe_png_post`` (aligned Before/After PNG bytes)
    - ``tif_bytes``, ``tif_filename`` (georeferenced GeoTIFF in ``target_crs``)
    """

    def _prog(pct: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    if not target_crs or str(target_crs).strip().lower() == "auto":
        from geodezija_crs import get_best_metric_crs

        c_lon = 0.5 * (float(bbox_4326[0]) + float(bbox_4326[2]))
        c_lat = 0.5 * (float(bbox_4326[1]) + float(bbox_4326[3]))
        target_crs, _ = get_best_metric_crs(c_lat, c_lon, mode="auto")

    sensor = select_sensor_for_disaster(disaster_type, event_date)
    is_sar = sensor == SENSOR_S1
    win_days = max(int(search_window_days), 25)

    # 1. Search Before / During / After windows
    _prog(0.10, "Iščem referenčni posnetek PRED dogodkom..." if lang == "sl" else "Searching PRE-event reference scene...")
    if is_sar:
        pre_start = event_date - timedelta(days=win_days)
        pre_end = event_date - timedelta(days=10)
        ev_start = event_date - timedelta(days=2)
        ev_end = event_date + timedelta(days=7)
        rec_start = event_date + timedelta(days=10)
        rec_end = event_date + timedelta(days=45)
        max_cc = 100.0
    else:
        pre_start = event_date - timedelta(days=max(win_days, 75))
        pre_end = event_date - timedelta(days=15)
        ev_start = event_date + timedelta(days=3)
        ev_end = event_date + timedelta(days=max(win_days, 60))
        rec_start = event_date - timedelta(days=2)
        rec_end = event_date + timedelta(days=5)
        max_cc = 20.0

    pre_items, pre_df = _search_window_with_fallback(
        sensor=sensor,
        bbox_4326=bbox_4326,
        start_date=pre_start,
        end_date=pre_end,
        max_cloud=max_cc,
        aoi_polygon_4326=aoi_polygon_4326,
    )
    pre_pick = _pick_best_scene(pre_items, pre_df, event_date - timedelta(days=20), is_sar=is_sar)
    if pre_pick is None:
        raise RuntimeError(
            f"Ni najdenega posnetka PRED dogodkom ({pre_start} – {pre_end}). Poskusite povečati iskalno okno."
            if lang == "sl"
            else f"No PRE-event scene found ({pre_start} – {pre_end}). Try increasing the search window."
        )

    _prog(0.28, "Iščem posnetek MED / PO dogodku..." if lang == "sl" else "Searching DURING / POST-event scene...")
    ev_items, ev_df = _search_window_with_fallback(
        sensor=sensor,
        bbox_4326=bbox_4326,
        start_date=ev_start,
        end_date=ev_end,
        max_cloud=max_cc,
        aoi_polygon_4326=aoi_polygon_4326,
    )
    ev_pick = _pick_best_scene(
        ev_items,
        ev_df,
        event_date if is_sar else event_date + timedelta(days=14),
        is_sar=is_sar,
    )
    if ev_pick is None:
        raise RuntimeError(
            f"Ni najdenega posnetka MED/PO dogodku ({ev_start} – {ev_end}). Poskusite povečati iskalno okno."
            if lang == "sl"
            else f"No DURING/POST-event scene found ({ev_start} – {ev_end}). Try increasing the search window."
        )

    # Optional 3rd scene metadata (Post-event recovery for floods / During-event date for wildfire)
    third_date_str: str | None = None
    try:
        opt_items, opt_df = _search_window_with_fallback(
            sensor=sensor,
            bbox_4326=bbox_4326,
            start_date=rec_start,
            end_date=rec_end,
            max_cloud=min(max_cc + 15.0, 40.0) if not is_sar else 100.0,
            aoi_polygon_4326=aoi_polygon_4326,
        )
        opt_pick = _pick_best_scene(opt_items, opt_df, rec_start + timedelta(days=5), is_sar=is_sar)
        if opt_pick is not None:
            third_date_str = str(opt_pick[1]["date_str"])
    except Exception:
        third_date_str = None

    pre_group, pre_row = pre_pick
    ev_group, ev_row = ev_pick
    date_pre = str(pre_row["date_str"])
    date_ev = str(ev_row["date_str"])

    # 2. Stream both scenes onto the exact same metric CRS grid (with same-day mosaicking!)
    _prog(0.48, f"Prenašam posnetek PRED ({date_pre}) v {target_crs}..." if lang == "sl" else f"Streaming PRE scene ({date_pre}) in {target_crs}...")
    if is_sar:
        load_bands = ["vv"]
    else:
        mask_band = "SCL" if sensor == SENSOR_S2 else "QA_PIXEL"
        if disaster_type == DISASTER_WILDFIRE:
            load_bands = ["B04", "B03", "B02", "B08", "B12", mask_band]
        else:
            load_bands = ["B04", "B03", "B02", "B08", mask_band]

    ds_pre = load_scene(
        sensor=sensor,
        item=pre_group,
        bbox_4326=bbox_4326,
        bands=load_bands,
        target_epsg=target_crs,
        resolution=int(resolution),
        aoi_polygon_4326=aoi_polygon_4326,
    )

    _prog(0.68, f"Prenašam posnetek MED/PO ({date_ev}) v {target_crs}..." if lang == "sl" else f"Streaming EVENT/POST scene ({date_ev}) in {target_crs}...")
    ds_ev = load_scene(
        sensor=sensor,
        item=ev_group,
        bbox_4326=bbox_4326,
        bands=load_bands,
        target_epsg=target_crs,
        resolution=int(resolution),
        aoi_polygon_4326=aoi_polygon_4326,
    )

    # 3. Compute before/after swipe RGB/SAR pair + disaster impact classification
    _prog(0.84, "Računam karto prizadetosti in površine [ha]..." if lang == "sl" else "Computing impact map and exact areas [ha]...")
    swipe_pre_png, swipe_post_png = render_disaster_swipe_pair(ds_pre, ds_ev, is_sar=is_sar)

    continuous_da: xr.DataArray | None = None
    if disaster_type == DISASTER_FLOOD:
        class_da, breakdown_df, summary = classify_flood_sar(
            ds_pre["VV"],
            ds_ev["VV"],
            threshold_db=float(flood_threshold_db),
            delta_db=float(flood_delta_db),
            resolution_m=float(resolution),
            lang=lang,
        )
    elif disaster_type == DISASTER_WILDFIRE:
        ds_pre_m, _ = apply_cloud_mask(ds_pre, sensor=sensor)
        ds_ev_m, _ = apply_cloud_mask(ds_ev, sensor=sensor)
        nbr_pre = compute_index(ds_pre_m, "NBR")
        nbr_post = compute_index(ds_ev_m, "NBR")
        class_da, continuous_da, breakdown_df, summary = classify_wildfire_dnbr(
            nbr_pre,
            nbr_post,
            resolution_m=float(resolution),
            lang=lang,
        )
    else:
        ds_pre_m, _ = apply_cloud_mask(ds_pre, sensor=sensor)
        ds_ev_m, _ = apply_cloud_mask(ds_ev, sensor=sensor)
        ndvi_pre = compute_index(ds_pre_m, "NDVI")
        ndvi_post = compute_index(ds_ev_m, "NDVI")
        class_da, continuous_da, breakdown_df, summary = classify_forest_dndvi(
            ndvi_pre,
            ndvi_post,
            resolution_m=float(resolution),
            lang=lang,
        )

    _prog(0.95, "Pripravljam GeoTIFF izvoz..." if lang == "sl" else "Preparing GeoTIFF export...")
    tif_bytes = export_to_geotiff_bytes(class_da, target_epsg=target_crs)
    epsg_num = str(target_crs).split(":")[-1]
    tif_filename = (
        f"Disaster_{disaster_type}_{date_pre}_to_{date_ev}_EPSG{epsg_num}_{int(resolution)}m.tif"
    )

    del ds_pre, ds_ev
    gc.collect()
    _prog(1.0, "Analiza zaključena." if lang == "sl" else "Analysis complete.")

    return {
        "disaster_type": disaster_type,
        "sensor": sensor,
        "crs": target_crs,
        "resolution_m": int(resolution),
        "event_date": event_date.isoformat(),
        "date_pre": date_pre,
        "date_event": date_ev,
        "date_optional": third_date_str,
        "tile_pre": str(pre_row.get("mgrs_tile", "")),
        "tile_event": str(ev_row.get("mgrs_tile", "")),
        "class_da": class_da,
        "continuous_da": continuous_da,
        "breakdown_df": breakdown_df,
        "summary": summary,
        "swipe_png_pre": swipe_pre_png,
        "swipe_png_post": swipe_post_png,
        "tif_bytes": tif_bytes,
        "tif_filename": tif_filename,
        "bbox_4326": list(bbox_4326),
        "polygon_4326": aoi_polygon_4326,
    }
