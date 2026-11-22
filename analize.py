"""Multi-sensor cloud/shadow masking, optical spectral indices, SAR analytics, and statistics.

Supports:
- Sentinel-2 L2A SCL cloud & shadow masking (`classes 0, 1, 3, 8, 9, 10`).
- Landsat Collection 2 Level-2 `QA_PIXEL` bitmasking (`(qa & 0b11110) != 0`).
- Harmonized optical spectral indices: NDVI, NDWI, MNDWI, NDBI, NBR.
- Sentinel-1 SAR RTC decibel analytics (`VV`, `VH`, `VV/VH` dB ratio, and
  threshold-based water/flood mapping `VV < threshold_db`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers .rio accessor on xarray objects
import xarray as xr

# Sentinel-2 L2A Scene Classification Layer (SCL) classes masked out as invalid/cloudy:
# 0: No Data, 1: Saturated/Defective, 3: Cloud Shadows,
# 8: Cloud Medium Prob., 9: Cloud High Prob., 10: Thin Cirrus
SCL_INVALID_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10)

# Landsat C2 L2 QA_PIXEL cloud/shadow bitmask:
# bit 1: Dilated Cloud, bit 2: Cirrus, bit 3: Cloud, bit 4: Cloud Shadow
LANDSAT_QA_CLOUD_MASK: int = 0b11110  # 30
LANDSAT_QA_FILL_MASK: int = 0b00001   # 1

SCL_CLASS_LABELS: dict[int, str] = {
    0: "No Data",
    1: "Saturated / Defective",
    2: "Dark Area / Cast Shadows",
    3: "Cloud Shadows",
    4: "Vegetation",
    5: "Not Vegetated",
    6: "Water",
    7: "Unclassified",
    8: "Cloud Medium Probability",
    9: "Cloud High Probability",
    10: "Thin Cirrus",
    11: "Snow / Ice",
}

SPECTRAL_INDICES: dict[str, dict[str, Any]] = {
    "NDVI": {
        "bands": ("B08", "B04"),
        "formula": "(B08 - B04) / (B08 + B04)",
        "cmap": "RdYlGn",
        "vmin": -0.2,
        "vmax": 0.9,
        "diff_cmap": "RdYlGn",
    },
    "NDWI": {
        "bands": ("B03", "B08"),
        "formula": "(B03 - B08) / (B03 + B08)",
        "cmap": "RdBu",
        "vmin": -0.6,
        "vmax": 0.6,
        "diff_cmap": "RdBu",
    },
    "MNDWI": {
        "bands": ("B03", "B11"),
        "formula": "(B03 - B11) / (B03 + B11)",
        "cmap": "RdBu",
        "vmin": -0.6,
        "vmax": 0.6,
        "diff_cmap": "RdBu",
    },
    "NDBI": {
        "bands": ("B11", "B08"),
        "formula": "(B11 - B08) / (B11 + B08)",
        "cmap": "RdYlBu_r",
        "vmin": -0.5,
        "vmax": 0.5,
        "diff_cmap": "RdBu_r",
    },
    "NBR": {
        "bands": ("B08", "B12"),
        "formula": "(B08 - B12) / (B08 + B12)",
        "cmap": "BrBG",
        "vmin": -0.5,
        "vmax": 0.8,
        "diff_cmap": "BrBG",
    },
    "NDSI": {
        "bands": ("B03", "B11"),
        "formula": "(B03 - B11) / (B03 + B11)",
        "cmap": "PuBu",
        "vmin": -0.2,
        "vmax": 0.9,
        "diff_cmap": "RdBu",
    },
}

SAR_METRICS: dict[str, dict[str, Any]] = {
    "VV": {
        "pols": ("vv",),
        "formula": "10 * log10(VV)",
        "cmap": "Greys_r",
        "vmin": -22.0,
        "vmax": -2.0,
        "diff_cmap": "RdBu",
        "units": "dB",
    },
    "VH": {
        "pols": ("vh",),
        "formula": "10 * log10(VH)",
        "cmap": "Greys_r",
        "vmin": -28.0,
        "vmax": -8.0,
        "diff_cmap": "RdBu",
        "units": "dB",
    },
    "VV_VH": {
        "pols": ("vv", "vh"),
        "formula": "VV[dB] - VH[dB]",
        "cmap": "BrBG",
        "vmin": 2.0,
        "vmax": 14.0,
        "diff_cmap": "RdBu",
        "units": "dB",
    },
}

# Optical bands & nominal central wavelengths [nm] for Spectral Signature Profiler
OPTICAL_SIGNATURE_BANDS: tuple[tuple[str, int, str], ...] = (
    ("B02", 490, "Blue"),
    ("B03", 560, "Green"),
    ("B04", 665, "Red"),
    ("B08", 842, "NIR"),
    ("B11", 1610, "SWIR-1"),
    ("B12", 2190, "SWIR-2"),
)

# 4-class physical land-cover definitions (clean cartographic palette)
LAND_COVER_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "class_key": "water",
        "class_id": 1,
        "rule": "NDWI > 0.1",
        "color": "#0369A1",
    },
    {
        "class_key": "built_bare",
        "class_id": 2,
        "rule": "NDWI <= 0.1 & NDVI < 0.2",
        "color": "#94A3B8",
    },
    {
        "class_key": "sparse_veg",
        "class_id": 3,
        "rule": "NDWI <= 0.1 & 0.2 <= NDVI < 0.5",
        "color": "#84CC16",
    },
    {
        "class_key": "dense_veg",
        "class_id": 4,
        "rule": "NDWI <= 0.1 & NDVI >= 0.5",
        "color": "#0F766E",
    },
)


def get_required_bands(index_name: str, include_scl: bool = True) -> list[str]:
    """Return the minimal list of Sentinel-2 bands needed for an index calculation."""
    key = index_name.strip().upper()
    if key not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unsupported spectral index {index_name!r}. "
            f"Supported indices: {', '.join(SPECTRAL_INDICES.keys())}."
        )
    bands = list(SPECTRAL_INDICES[key]["bands"])
    if include_scl and "SCL" not in bands:
        bands.append("SCL")
    return bands


def get_required_bands_for_sensor(
    sensor: str,
    metric_name: str,
    include_cloud_mask: bool = True,
) -> list[str]:
    """Return the minimal band/polarization list for a given sensor and index/metric."""
    key = metric_name.strip().upper()
    if sensor == "sentinel-1-rtc":
        if key not in SAR_METRICS:
            raise ValueError(
                f"Unsupported SAR metric {metric_name!r}. "
                f"Supported SAR metrics: {', '.join(SAR_METRICS.keys())}."
            )
        return list(SAR_METRICS[key]["pols"])

    if key not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unsupported spectral index {metric_name!r}. "
            f"Supported indices: {', '.join(SPECTRAL_INDICES.keys())}."
        )
    bands = list(SPECTRAL_INDICES[key]["bands"])
    if include_cloud_mask:
        mask_band = "QA_PIXEL" if sensor == "landsat-c2-l2" else "SCL"
        if mask_band not in bands:
            bands.append(mask_band)
    return bands


def apply_scl_mask(
    ds: xr.Dataset,
    mask_clouds: bool = True,
) -> tuple[xr.Dataset, float]:
    """Mask cloudy, shadowed, and defective pixels using Sentinel-2 L2A SCL band.

    Masks SCL classes ``{0, 1, 3, 8, 9, 10}`` to ``np.nan`` across all optical
    data variables while keeping ``SCL`` intact as ``uint8``.
    """
    optical_vars = [name for name in ds.data_vars if str(name) != "SCL"]
    if not optical_vars:
        return ds, 0.0

    ref_shape = ds[optical_vars[0]].shape
    total_pixels = int(np.prod(ref_shape))
    if total_pixels == 0:
        return ds, 0.0

    optical_finite = np.ones(ref_shape, dtype=bool)
    for var in optical_vars:
        optical_finite &= np.isfinite(ds[var].values)

    if not mask_clouds or "SCL" not in ds.data_vars:
        valid_pct = float(100.0 * np.count_nonzero(optical_finite) / total_pixels)
        return ds, round(valid_pct, 2)

    scl_vals = ds["SCL"].values
    scl_clear = ~np.isin(scl_vals, SCL_INVALID_CLASSES)
    valid_mask = optical_finite & scl_clear
    valid_pct = float(100.0 * np.count_nonzero(valid_mask) / total_pixels)

    src_crs = ds.rio.crs
    masked_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        da = ds[var]
        if str(var) == "SCL":
            masked_vars[str(var)] = da
        else:
            arr = da.values.astype(np.float32, copy=True)
            arr[~scl_clear] = np.nan
            out_da = xr.DataArray(
                arr,
                dims=da.dims,
                coords=da.coords,
                name=da.name,
                attrs=dict(da.attrs),
            )
            if src_crs is not None:
                out_da = out_da.rio.write_crs(src_crs, inplace=False)
            out_da = out_da.rio.write_nodata(np.nan, inplace=False)
            masked_vars[str(var)] = out_da

    masked_ds = xr.Dataset(
        data_vars=masked_vars,
        coords=ds.coords,
        attrs={**dict(ds.attrs), "scl_mask_applied": True, "valid_pixel_pct": round(valid_pct, 2)},
    )
    if src_crs is not None:
        masked_ds = masked_ds.rio.write_crs(src_crs, inplace=False)

    return masked_ds, round(valid_pct, 2)


def apply_landsat_qa_mask(
    ds: xr.Dataset,
    mask_clouds: bool = True,
) -> tuple[xr.Dataset, float]:
    """Mask cloudy, cirrus, shadow, and fill pixels using Landsat C2 L2 ``QA_PIXEL``.

    Masks pixels where bit 1 (Dilated Cloud), bit 2 (Cirrus), bit 3 (Cloud),
    or bit 4 (Cloud Shadow) is set (``(qa & 0b11110) != 0``), as well as Fill
    bit 0 (``(qa & 0b00001) != 0``), replacing optical reflectance with ``np.nan``.
    """
    qa_key = "QA_PIXEL" if "QA_PIXEL" in ds.data_vars else ("qa_pixel" if "qa_pixel" in ds.data_vars else None)
    optical_vars = [name for name in ds.data_vars if str(name) != qa_key]
    if not optical_vars:
        return ds, 0.0

    ref_shape = ds[optical_vars[0]].shape
    total_pixels = int(np.prod(ref_shape))
    if total_pixels == 0:
        return ds, 0.0

    optical_finite = np.ones(ref_shape, dtype=bool)
    for var in optical_vars:
        optical_finite &= np.isfinite(ds[var].values)

    if not mask_clouds or qa_key is None:
        valid_pct = float(100.0 * np.count_nonzero(optical_finite) / total_pixels)
        return ds, round(valid_pct, 2)

    qa_vals = ds[qa_key].values.astype(np.uint16, copy=False)
    is_cloudy_or_fill = ((qa_vals & np.uint16(LANDSAT_QA_CLOUD_MASK)) != 0) | (
        (qa_vals & np.uint16(LANDSAT_QA_FILL_MASK)) != 0
    )
    qa_clear = ~is_cloudy_or_fill
    valid_mask = optical_finite & qa_clear
    valid_pct = float(100.0 * np.count_nonzero(valid_mask) / total_pixels)

    src_crs = ds.rio.crs
    masked_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        da = ds[var]
        if str(var) == qa_key:
            masked_vars[str(var)] = da
        else:
            arr = da.values.astype(np.float32, copy=True)
            arr[~qa_clear] = np.nan
            out_da = xr.DataArray(
                arr,
                dims=da.dims,
                coords=da.coords,
                name=da.name,
                attrs=dict(da.attrs),
            )
            if src_crs is not None:
                out_da = out_da.rio.write_crs(src_crs, inplace=False)
            out_da = out_da.rio.write_nodata(np.nan, inplace=False)
            masked_vars[str(var)] = out_da

    masked_ds = xr.Dataset(
        data_vars=masked_vars,
        coords=ds.coords,
        attrs={**dict(ds.attrs), "qa_mask_applied": True, "valid_pixel_pct": round(valid_pct, 2)},
    )
    if src_crs is not None:
        masked_ds = masked_ds.rio.write_crs(src_crs, inplace=False)

    return masked_ds, round(valid_pct, 2)


def apply_cloud_mask(
    ds: xr.Dataset,
    mask_clouds: bool = True,
    sensor: str | None = None,
    sensor_id: str | None = None,
    **_kwargs: Any,
) -> tuple[xr.Dataset, float]:
    """Unified multi-sensor cloud mask dispatcher (Sentinel-2 SCL, Landsat QA_PIXEL, or SAR).

    Accepts optional ``sensor`` / ``sensor_id`` arguments for full compatibility across
    all modules (``nesrece.py``, ``ai_pomocnik.py``, ``gnss_plan.py``, ``casovne_vrste.py``, ``app.py``).
    """
    eff_sensor = str(sensor or sensor_id or "").strip().lower()
    if eff_sensor == "sentinel-1-rtc":
        mask_clouds = False

    if "SCL" in ds.data_vars and eff_sensor != "landsat-c2-l2":
        return apply_scl_mask(ds, mask_clouds=mask_clouds)
    if "QA_PIXEL" in ds.data_vars or "qa_pixel" in ds.data_vars:
        return apply_landsat_qa_mask(ds, mask_clouds=mask_clouds)

    # Radar (Sentinel-1) or dataset without QA band: report finite coverage
    data_vars = list(ds.data_vars)
    if not data_vars:
        return ds, 0.0
    ref_shape = ds[data_vars[0]].shape
    total_pixels = int(np.prod(ref_shape))
    if total_pixels == 0:
        return ds, 0.0
    finite_mask = np.ones(ref_shape, dtype=bool)
    for var in data_vars:
        finite_mask &= np.isfinite(ds[var].values)
    valid_pct = float(100.0 * np.count_nonzero(finite_mask) / total_pixels)
    return ds, round(valid_pct, 2)


def compute_index(ds: xr.Dataset, index_name: str) -> xr.DataArray:
    """Compute a normalized-difference spectral index clipped to ``[-1.0, 1.0]``.

    Works identically for both Sentinel-2 L2A and harmonized Landsat C2 L2 Datasets.
    """
    key = index_name.strip().upper()
    if key not in SPECTRAL_INDICES:
        raise ValueError(
            f"Unsupported spectral index {index_name!r}. "
            f"Choose one of: {', '.join(SPECTRAL_INDICES.keys())}."
        )

    spec = SPECTRAL_INDICES[key]
    b1, b2 = spec["bands"]
    if b1 not in ds.data_vars or b2 not in ds.data_vars:
        raise KeyError(
            f"Dataset is missing required bands ({b1}, {b2}) for index {key}. "
            f"Available bands: {list(ds.data_vars)}."
        )

    da1 = ds[b1]
    da2 = ds[b2]
    arr1 = da1.values.astype(np.float32, copy=False)
    arr2 = da2.values.astype(np.float32, copy=False)

    num = arr1 - arr2
    denom = arr1 + arr2
    valid = np.isfinite(num) & np.isfinite(denom) & (denom != 0.0)

    out = np.full(arr1.shape, np.nan, dtype=np.float32)
    np.divide(num, denom, out=out, where=valid)
    np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)

    index_da = xr.DataArray(
        out,
        dims=da1.dims,
        coords=da1.coords,
        name=key,
        attrs={
            "index_name": key,
            "formula": spec["formula"],
            "bands": f"{b1}, {b2}",
            "units": "dimensionless",
            "long_name": f"{key}",
            "sensor": ds.attrs.get("sensor", ""),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
            "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
            "resolution_m": ds.attrs.get("resolution_m", 10.0),
        },
    )

    crs = ds.rio.crs or da1.rio.crs
    if crs is not None:
        index_da = index_da.rio.write_crs(crs, inplace=False)
    index_da = index_da.rio.write_nodata(np.nan, inplace=False)
    return index_da


def extract_sar_metric(ds: xr.Dataset, metric_name: str) -> xr.DataArray:
    """Extract a Sentinel-1 SAR decibel layer: ``VV``, ``VH``, or ``VV_VH`` (VV - VH in dB)."""
    key = metric_name.strip().upper()
    if key not in SAR_METRICS:
        raise ValueError(
            f"Unsupported SAR metric {metric_name!r}. "
            f"Choose one of: {', '.join(SAR_METRICS.keys())}."
        )

    crs = ds.rio.crs
    if key in ("VV", "VH"):
        if key not in ds.data_vars:
            raise KeyError(f"SAR dataset missing polarization {key}. Available: {list(ds.data_vars)}")
        da = ds[key].astype(np.float32, copy=False)
        da = da.assign_attrs(
            {
                **dict(da.attrs),
                "index_name": key,
                "units": "dB",
                "sensor": ds.attrs.get("sensor", "sentinel-1-rtc"),
                "scene_id": ds.attrs.get("scene_id", ""),
                "datetime": ds.attrs.get("datetime", ""),
                "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
                "resolution_m": ds.attrs.get("resolution_m", 10.0),
            }
        )
        if crs is not None:
            da = da.rio.write_crs(crs, inplace=False)
        da = da.rio.write_nodata(np.nan, inplace=False)
        return da

    # VV / VH linear ratio in dB equals VV_dB - VH_dB
    if "VV" not in ds.data_vars or "VH" not in ds.data_vars:
        raise KeyError(f"VV_VH metric requires both VV and VH. Available: {list(ds.data_vars)}")

    vv = ds["VV"].values.astype(np.float32, copy=False)
    vh = ds["VH"].values.astype(np.float32, copy=False)
    valid = np.isfinite(vv) & np.isfinite(vh)
    ratio_db = np.full(vv.shape, np.nan, dtype=np.float32)
    np.subtract(vv, vh, out=ratio_db, where=valid)

    ratio_da = xr.DataArray(
        ratio_db,
        dims=ds["VV"].dims,
        coords=ds["VV"].coords,
        name="VV_VH",
        attrs={
            "index_name": "VV_VH",
            "formula": "VV[dB] - VH[dB]",
            "units": "dB",
            "long_name": "Sentinel-1 VV/VH Backscatter Ratio [dB]",
            "sensor": ds.attrs.get("sensor", "sentinel-1-rtc"),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
            "mgrs_tile": ds.attrs.get("mgrs_tile", ""),
            "resolution_m": ds.attrs.get("resolution_m", 10.0),
        },
    )
    if crs is not None:
        ratio_da = ratio_da.rio.write_crs(crs, inplace=False)
    ratio_da = ratio_da.rio.write_nodata(np.nan, inplace=False)
    return ratio_da


def compute_sar_water_mask(
    ds: xr.Dataset,
    threshold_db: float = -16.0,
    polarization: str = "VV",
    resolution_m: float = 10.0,
) -> tuple[xr.DataArray, dict[str, float | int]]:
    """Classify smooth water / flood surfaces from SAR backscatter (``pol < threshold_db``).

    Returns ``(mask_da, stats)`` where ``mask_da`` is a ``uint8`` georeferenced
    DataArray (`1` = Water / Flood, `0` = Land / Other, `255` = NoData) ready
    for visualization and GeoTIFF export, and ``stats`` includes exact water
    area in hectares (``water_area_ha``).
    """
    pol_key = polarization.strip().upper()
    if pol_key not in ds.data_vars:
        raise KeyError(f"Polarization {pol_key!r} not in dataset. Available: {list(ds.data_vars)}")
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be strictly positive, got {resolution_m}.")

    pol_da = ds[pol_key]
    vals = pol_da.values.astype(np.float32, copy=False)
    valid = np.isfinite(vals)
    is_water = valid & (vals < np.float32(threshold_db))

    mask_arr = np.full(vals.shape, 255, dtype=np.uint8)
    mask_arr[valid] = 0
    mask_arr[is_water] = 1

    total_pixels = int(vals.size)
    valid_pixels = int(np.count_nonzero(valid))
    water_pixels = int(np.count_nonzero(is_water))
    px_area_ha = (float(resolution_m) ** 2) / 10_000.0

    valid_area_ha = valid_pixels * px_area_ha
    water_area_ha = water_pixels * px_area_ha
    valid_pct = (100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    water_pct = (100.0 * water_pixels / valid_pixels) if valid_pixels > 0 else 0.0

    mask_da = xr.DataArray(
        mask_arr,
        dims=pol_da.dims,
        coords=pol_da.coords,
        name=f"SAR_WATER_{pol_key}",
        attrs={
            "long_name": f"Sentinel-1 Water/Flood Mask ({pol_key} < {threshold_db:.1f} dB)",
            "threshold_db": float(threshold_db),
            "polarization": pol_key,
            "classes": "0=Land, 1=Water/Flood, 255=NoData",
            "resolution_m": float(resolution_m),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
        },
    )
    crs = ds.rio.crs or pol_da.rio.crs
    if crs is not None:
        mask_da = mask_da.rio.write_crs(crs, inplace=False)
    mask_da = mask_da.rio.write_nodata(255, inplace=False)

    stats: dict[str, float | int] = {
        "threshold_db": float(threshold_db),
        "water_pixels": water_pixels,
        "water_area_ha": round(water_area_ha, 3),
        "water_pct": round(water_pct, 2),
        "valid_pixels": valid_pixels,
        "valid_area_ha": round(valid_area_ha, 3),
        "valid_pct": round(valid_pct, 2),
    }
    return mask_da, stats


def compute_index_stats(
    index_da: xr.DataArray,
    resolution_m: float,
) -> dict[str, float | int]:
    """Compute descriptive statistics and exact valid area in hectares for an index or SAR raster."""
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be strictly positive, got {resolution_m}.")

    vals = index_da.values
    total_pixels = int(vals.size)
    valid_vals = vals[np.isfinite(vals)]
    valid_pixels = int(valid_vals.size)
    valid_pct = float(100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    valid_area_ha = float(valid_pixels * (float(resolution_m) ** 2) / 10_000.0)

    if valid_pixels == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "valid_pixels": 0,
            "valid_pct": 0.0,
            "valid_area_ha": 0.0,
        }

    p10, median, p90 = np.percentile(valid_vals, [10.0, 50.0, 90.0])
    return {
        "min": float(np.min(valid_vals)),
        "max": float(np.max(valid_vals)),
        "mean": float(np.mean(valid_vals)),
        "median": float(median),
        "std": float(np.std(valid_vals)),
        "p10": float(p10),
        "p90": float(p90),
        "valid_pixels": valid_pixels,
        "valid_pct": round(valid_pct, 2),
        "valid_area_ha": round(valid_area_ha, 3),
    }


def extract_spectral_signature(
    ds: xr.Dataset,
    x_coord: float,
    y_coord: float,
) -> pd.DataFrame:
    """Extract surface reflectance ``[0, 1]`` at the nearest pixel to ``(x_coord, y_coord)``.

    Works for both Sentinel-2 L2A and harmonized Landsat C2 L2 Datasets across
    ``B02 (490 nm)``, ``B03 (560 nm)``, ``B04 (665 nm)``, ``B08 (842 nm)``,
    ``B11 (1610 nm)``, and ``B12 (2190 nm)``.
    """
    x_vals = np.asarray(ds.coords["x"].values, dtype=np.float64)
    y_vals = np.asarray(ds.coords["y"].values, dtype=np.float64)
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Dataset has empty spatial coordinates.")

    ix = int(np.argmin(np.abs(x_vals - float(x_coord))))
    iy = int(np.argmin(np.abs(y_vals - float(y_coord))))
    actual_x = float(x_vals[ix])
    actual_y = float(y_vals[iy])

    rows: list[dict[str, Any]] = []
    for band_code, wl_nm, band_name in OPTICAL_SIGNATURE_BANDS:
        if band_code not in ds.data_vars:
            continue
        raw_val = float(ds[band_code].isel(y=iy, x=ix).values)
        rows.append(
            {
                "band": band_code,
                "wavelength_nm": int(wl_nm),
                "band_name": band_name,
                "reflectance": raw_val if np.isfinite(raw_val) else np.nan,
            }
        )

    df = pd.DataFrame(rows, columns=["band", "wavelength_nm", "band_name", "reflectance"])
    df.attrs["actual_x"] = actual_x
    df.attrs["actual_y"] = actual_y
    df.attrs["ix"] = ix
    df.attrs["iy"] = iy
    return df


def compute_quick_land_cover(
    ds: xr.Dataset,
    resolution_m: float,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Classify an optical (cloud-masked) scene into 4 intuitive physical land-cover classes.

    Classes:
    - ``1`` (``water``): ``NDWI > 0.1``
    - ``2`` (``built_bare``): ``NDWI <= 0.1`` and ``NDVI < 0.2``
    - ``3`` (``sparse_veg``): ``NDWI <= 0.1`` and ``0.2 <= NDVI < 0.5``
    - ``4`` (``dense_veg``): ``NDWI <= 0.1`` and ``NDVI >= 0.5``
    - ``255``: NoData / Cloud-masked pixels
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be strictly positive, got {resolution_m}.")

    ndvi_da = compute_index(ds, "NDVI")
    ndwi_da = compute_index(ds, "NDWI")

    ndvi = ndvi_da.values
    ndwi = ndwi_da.values
    valid = np.isfinite(ndvi) & np.isfinite(ndwi)

    lc_arr = np.full(ndvi.shape, 255, dtype=np.uint8)
    is_water = valid & (ndwi > 0.1)
    not_water = valid & (~is_water)
    is_built = not_water & (ndvi < 0.2)
    is_sparse = not_water & (ndvi >= 0.2) & (ndvi < 0.5)
    is_dense = not_water & (ndvi >= 0.5)

    lc_arr[is_water] = 1
    lc_arr[is_built] = 2
    lc_arr[is_sparse] = 3
    lc_arr[is_dense] = 4

    valid_pixels = int(np.count_nonzero(valid))
    px_area_ha = (float(resolution_m) ** 2) / 10_000.0

    counts = {
        "water": int(np.count_nonzero(is_water)),
        "built_bare": int(np.count_nonzero(is_built)),
        "sparse_veg": int(np.count_nonzero(is_sparse)),
        "dense_veg": int(np.count_nonzero(is_dense)),
    }

    rows: list[dict[str, Any]] = []
    for spec in LAND_COVER_CLASSES:
        ckey = spec["class_key"]
        px = counts[ckey]
        area_ha = px * px_area_ha
        share_pct = (100.0 * px / valid_pixels) if valid_pixels > 0 else 0.0
        rows.append(
            {
                "class_key": ckey,
                "class_id": spec["class_id"],
                "rule": spec["rule"],
                "color": spec["color"],
                "pixels": px,
                "area_ha": round(area_ha, 3),
                "share_pct": round(share_pct, 2),
            }
        )

    lc_da = xr.DataArray(
        lc_arr,
        dims=ndvi_da.dims,
        coords=ndvi_da.coords,
        name="LAND_COVER",
        attrs={
            "long_name": "Quick Physical Land-Cover Classification",
            "classes": "1=Water, 2=Built-up/Bare, 3=Sparse Veg, 4=Dense Veg, 255=NoData",
            "resolution_m": float(resolution_m),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
        },
    )
    crs = ds.rio.crs or ndvi_da.rio.crs
    if crs is not None:
        lc_da = lc_da.rio.write_crs(crs, inplace=False)
    lc_da = lc_da.rio.write_nodata(255, inplace=False)

    summary_df = pd.DataFrame(rows)
    summary_df.attrs["valid_pixels"] = valid_pixels
    summary_df.attrs["valid_area_ha"] = round(valid_pixels * px_area_ha, 3)
    return lc_da, summary_df


def compute_snow_glacier_stats(
    ds: xr.Dataset,
    dem_da: xr.DataArray | None = None,
    ndsi_thresh: float = 0.40,
    nir_thresh: float = 0.11,
    resolution_m: float | None = None,
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Classify Snow / Ice / Glacier extent via ESA/NASA NDSI + NIR mask and optional DEM snowline.

    Uses the standard ESA/NASA optical cryosphere rule:
    ``is_snow = (NDSI >= ndsi_thresh) & (B08 > nir_thresh)`` (when ``B08`` is present),
    where the ``NIR > 0.11`` condition cleanly separates bright snow/ice from dark water bodies.

    When ``dem_da`` (e.g., Copernicus DEM 30 m) is supplied, computes the glaciological
    snowline altitude (10th percentile elevation of snow/ice pixels), plus median and max
    snow elevations [m a.s.l.].

    Returns
    -------
    tuple[xr.DataArray, dict[str, Any]]
        ``(snow_mask_da, stats)`` where ``snow_mask_da`` is a ``uint8`` georeferenced raster
        (``1`` = Snow/Ice/Glacier, ``0`` = Snow-Free Land/Water, ``255`` = NoData/Cloud)
        and ``stats`` contains exact ``snow_area_ha``, ``snow_area_km2``, ``snow_pct``,
        and DEM elevation statistics (``snowline_p10_m``, ``snow_median_elev_m``, ``snow_max_elev_m``).
    """
    ndsi_da = compute_index(ds, "NDSI")
    ndsi_vals = ndsi_da.values.astype(np.float32, copy=False)

    # Infer exact metric pixel size from argument, dataset attrs, or coordinates
    if resolution_m is not None and float(resolution_m) > 0:
        res_m = float(resolution_m)
    else:
        attr_res = ds.attrs.get("resolution_m")
        if attr_res is not None and float(attr_res) > 0:
            res_m = float(attr_res)
        else:
            x_c = np.asarray(ndsi_da.coords["x"].values, dtype=np.float64)
            res_m = float(abs(np.median(np.diff(x_c)))) if x_c.size > 1 else 10.0

    has_nir = "B08" in ds.data_vars
    if has_nir:
        nir_vals = ds["B08"].values.astype(np.float32, copy=False)
        valid = np.isfinite(ndsi_vals) & np.isfinite(nir_vals)
        is_snow = valid & (ndsi_vals >= np.float32(ndsi_thresh)) & (nir_vals > np.float32(nir_thresh))
    else:
        valid = np.isfinite(ndsi_vals)
        is_snow = valid & (ndsi_vals >= np.float32(ndsi_thresh))

    mask_arr = np.full(ndsi_vals.shape, 255, dtype=np.uint8)
    mask_arr[valid] = 0
    mask_arr[is_snow] = 1

    total_pixels = int(ndsi_vals.size)
    valid_pixels = int(np.count_nonzero(valid))
    snow_pixels = int(np.count_nonzero(is_snow))
    px_area_ha = (res_m ** 2) / 10_000.0
    snow_area_ha = snow_pixels * px_area_ha
    snow_area_km2 = snow_area_ha / 100.0
    valid_area_ha = valid_pixels * px_area_ha
    valid_pct = (100.0 * valid_pixels / total_pixels) if total_pixels > 0 else 0.0
    snow_pct = (100.0 * snow_pixels / valid_pixels) if valid_pixels > 0 else 0.0

    # Snowline elevation via Copernicus DEM 30 m (if provided)
    dem_available = False
    snowline_p10_m: float | None = None
    snow_median_elev_m: float | None = None
    snow_max_elev_m: float | None = None
    snow_min_elev_m: float | None = None

    if dem_da is not None and snow_pixels > 0:
        try:
            from rasterio.enums import Resampling

            dem_aligned = dem_da
            if dem_aligned.shape != ndsi_da.shape or (
                dem_aligned.rio.crs is not None
                and ndsi_da.rio.crs is not None
                and dem_aligned.rio.crs != ndsi_da.rio.crs
            ):
                dem_aligned = dem_aligned.rio.reproject_match(
                    ndsi_da,
                    resampling=Resampling.bilinear,
                    nodata=np.nan,
                )
            dem_vals = dem_aligned.values.astype(np.float32, copy=False)
            snow_elevs = dem_vals[is_snow & np.isfinite(dem_vals) & (dem_vals > -500.0)]
            if snow_elevs.size > 0:
                p10, med, mx, mn = (
                    float(np.percentile(snow_elevs, 10.0)),
                    float(np.median(snow_elevs)),
                    float(np.max(snow_elevs)),
                    float(np.min(snow_elevs)),
                )
                dem_available = True
                snowline_p10_m = round(p10, 1)
                snow_median_elev_m = round(med, 1)
                snow_max_elev_m = round(mx, 1)
                snow_min_elev_m = round(mn, 1)
        except Exception:
            pass

    rule_str = (
        f"NDSI >= {ndsi_thresh:.2f} & B08 > {nir_thresh:.2f}"
        if has_nir
        else f"NDSI >= {ndsi_thresh:.2f}"
    )
    snow_mask_da = xr.DataArray(
        mask_arr,
        dims=ndsi_da.dims,
        coords=ndsi_da.coords,
        name="SNOW_ICE_MASK",
        attrs={
            "long_name": f"Snow / Ice / Glacier Mask ({rule_str})",
            "ndsi_thresh": float(ndsi_thresh),
            "nir_thresh": float(nir_thresh),
            "nir_water_filter": bool(has_nir),
            "classes": "0=Snow-Free, 1=Snow/Ice/Glacier, 255=NoData",
            "resolution_m": float(res_m),
            "scene_id": ds.attrs.get("scene_id", ""),
            "datetime": ds.attrs.get("datetime", ""),
        },
    )
    crs = ds.rio.crs or ndsi_da.rio.crs
    if crs is not None:
        snow_mask_da = snow_mask_da.rio.write_crs(crs, inplace=False)
    snow_mask_da = snow_mask_da.rio.write_nodata(255, inplace=False)

    stats: dict[str, Any] = {
        "ndsi_thresh": float(ndsi_thresh),
        "nir_thresh": float(nir_thresh),
        "nir_water_filter": bool(has_nir),
        "resolution_m": float(res_m),
        "snow_pixels": snow_pixels,
        "snow_area_ha": round(snow_area_ha, 3),
        "snow_area_km2": round(snow_area_km2, 4),
        "snow_pct": round(snow_pct, 2),
        "valid_pixels": valid_pixels,
        "valid_area_ha": round(valid_area_ha, 3),
        "valid_pct": round(valid_pct, 2),
        "dem_available": dem_available,
        "snowline_p10_m": snowline_p10_m,
        "snow_median_elev_m": snow_median_elev_m,
        "snow_max_elev_m": snow_max_elev_m,
        "snow_min_elev_m": snow_min_elev_m,
    }
    return snow_mask_da, stats


