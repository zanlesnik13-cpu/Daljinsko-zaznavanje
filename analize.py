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
) -> tuple[xr.Dataset, float]:
    """Unified multi-sensor cloud mask dispatcher (Sentinel-2 SCL, Landsat QA_PIXEL, or SAR)."""
    if "SCL" in ds.data_vars:
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
