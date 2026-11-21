"""Geodetic Cartography & Print Layout Engine (QGIS Print Layout-inspired).

Renders publication-grade metric maps strictly in the raster's true projected CRS
(D96/TM ``EPSG:3794`` or UTM 33N ``EPSG:32633``) with 1:1 equal metric aspect ratio
(zero Web Mercator distortion).

Features:
- 3-band optical RGB/NIR/SWIR, 2-band SAR (VV+VH), continuous single-band indices
  (NDVI, NDWI, MNDWI, NDBI, NBR, Change Detection Δ, SAR dB), and discrete masks
  (SAR Water / Flood Mask).
- Toggleable metric coordinate grid (auto or manual spacing, light/dark),
  exact alternating black-and-white geodetic scale bar + numerical scale (M ≈ 1 : N),
  minimalist cartographic North arrow, AOI polygon boundary overlay, scientific
  colorbar / band-combination legend, title header, and geodetic metadata footer.
- Exports 150 DPI preview PNG, 300 DPI high-resolution PNG, and 300 DPI vector-text PDF
  with immediate ``plt.close(fig)`` + ``gc.collect()`` cleanup for Streamlit Cloud RAM.
"""

from __future__ import annotations

import gc
import io
import math
from datetime import date
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform

from geodezija_crs import CRS_D96TM, CRS_UTM33N, CRS_WGS84, EPSG_D96TM, EPSG_UTM33N, normalize_crs

# ---------------------------------------------------------------------------
# Constants: Page Formats, Grid Steps, Optical Wavelengths
# ---------------------------------------------------------------------------
MM_TO_INCH: float = 1.0 / 25.4

PAGE_FORMATS: dict[str, dict[str, Any]] = {
    "a4_landscape": {
        "width_mm": 297.0,
        "height_mm": 210.0,
        "figsize": (297.0 * MM_TO_INCH, 210.0 * MM_TO_INCH),
        "orientation": "landscape",
    },
    "a4_portrait": {
        "width_mm": 210.0,
        "height_mm": 297.0,
        "figsize": (210.0 * MM_TO_INCH, 297.0 * MM_TO_INCH),
        "orientation": "portrait",
    },
    "square": {
        "width_mm": 210.0,
        "height_mm": 210.0,
        "figsize": (210.0 * MM_TO_INCH, 210.0 * MM_TO_INCH),
        "orientation": "square",
    },
}

NICE_GRID_INTERVALS_M: tuple[int, ...] = (
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
)

NICE_SCALEBAR_LENGTHS_M: tuple[int, ...] = (
    50,
    100,
    200,
    250,
    500,
    1000,
    2000,
    2500,
    5000,
    10000,
    20000,
    25000,
    50000,
)

BAND_WAVELENGTH_LABELS: dict[str, str] = {
    "B02": "B02 (490 nm — Blue)",
    "B03": "B03 (560 nm — Green)",
    "B04": "B04 (665 nm — Red)",
    "B05": "B05 (705 nm — Red Edge)",
    "B08": "B08 (842 nm — NIR)",
    "B11": "B11 (1610 nm — SWIR 1)",
    "B12": "B12 (2190 nm — SWIR 2)",
    "VV": "VV [dB]",
    "VH": "VH [dB]",
    "VV-VH": "VV − VH [dB]",
}

# Palette (strictly cartographic teal / slate, no purple/violet)
COLOR_TEAL = "#0F766E"
COLOR_SLATE_900 = "#0F172A"
COLOR_SLATE_700 = "#334155"
COLOR_SLATE_500 = "#64748B"
COLOR_SLATE_300 = "#CBD5E1"
COLOR_BG_LIGHT = "#F8FAFC"
COLOR_NODATA = "#EEF2F6"


# ---------------------------------------------------------------------------
# Metric Calculation Helpers
# ---------------------------------------------------------------------------
def format_metric_number(value: float | int) -> str:
    """Format an integer/metric coordinate with narrow space thousands separators."""
    return f"{int(round(value)):,}".replace(",", " ")


def compute_nice_grid_spacing(span_m: float, target_lines: int = 5) -> int:
    """Select a clean round metric grid interval in meters for a given map span."""
    span = max(float(span_m), 100.0)
    raw = span / max(float(target_lines), 2.0)
    for step in NICE_GRID_INTERVALS_M:
        if step >= raw * 0.85:
            return step
    return NICE_GRID_INTERVALS_M[-1]


def compute_nice_scalebar_length(width_m: float, target_fraction: float = 0.22) -> int:
    """Select a clean round metric scale-bar length (~20-25% of map width)."""
    target = max(float(width_m) * float(target_fraction), 50.0)
    best = NICE_SCALEBAR_LENGTHS_M[0]
    best_dist = abs(best - target)
    for cand in NICE_SCALEBAR_LENGTHS_M:
        # Never exceed 35% of map width
        if cand > width_m * 0.35:
            break
        dist = abs(cand - target)
        if dist < best_dist:
            best = cand
            best_dist = dist
    return best


def compute_nice_scale_denom(ground_width_m: float, paper_width_m: float) -> int:
    """Compute approximate printed numerical scale denominator (e.g. 25 000)."""
    if paper_width_m <= 0.0 or ground_width_m <= 0.0:
        return 10000
    raw = ground_width_m / paper_width_m
    if raw < 1000.0:
        return max(100, int(round(raw / 50.0) * 50))
    if raw < 10000.0:
        return int(round(raw / 250.0) * 250)
    if raw < 50000.0:
        return int(round(raw / 500.0) * 500)
    if raw < 200000.0:
        return int(round(raw / 1000.0) * 1000)
    return int(round(raw / 5000.0) * 5000)


def _format_scalebar_label(meters: float) -> str:
    if meters == 0:
        return "0"
    if meters >= 1000.0:
        km = meters / 1000.0
        if abs(km - round(km)) < 1e-6:
            return f"{int(round(km))} km"
        return f"{km:.1f} km"
    return f"{int(round(meters))} m"


def _crs_display_info(crs_spec: int | str | CRS, lang: str = "sl") -> tuple[str, str, str]:
    """Return ``(crs_title, x_axis_label, y_axis_label)`` for a projected CRS."""
    crs_obj = normalize_crs(crs_spec)
    epsg = crs_obj.to_epsg()
    if epsg == EPSG_D96TM:
        return ("D96/TM (EPSG:3794)", "e [m]", "n [m]")
    if epsg == EPSG_UTM33N:
        return ("UTM 33N (EPSG:32633)", "E [m]", "N [m]")
    label = f"EPSG:{epsg}" if epsg else crs_obj.name
    return (label, "X [m]", "Y [m]")


def _extract_extent_and_orientation(
    obj: xr.Dataset | xr.DataArray,
) -> tuple[float, float, float, float, bool]:
    """Return exact pixel-edge ``(min_e, max_e, min_n, max_n, y_is_ascending)``."""
    x_vals = np.asarray(obj.coords["x"].values, dtype=np.float64)
    y_vals = np.asarray(obj.coords["y"].values, dtype=np.float64)
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Raster has empty spatial coordinates.")

    dx = float(abs(np.median(np.diff(x_vals)))) if x_vals.size > 1 else 10.0
    dy = float(abs(np.median(np.diff(y_vals)))) if y_vals.size > 1 else 10.0
    min_e = float(np.min(x_vals)) - 0.5 * dx
    max_e = float(np.max(x_vals)) + 0.5 * dx
    min_n = float(np.min(y_vals)) - 0.5 * dy
    max_n = float(np.max(y_vals)) + 0.5 * dy
    y_ascending = bool(y_vals.size > 1 and y_vals[0] < y_vals[-1])
    return min_e, max_e, min_n, max_n, y_ascending


def _stretch_channel_01(
    arr: np.ndarray,
    *,
    use_percentile_stretch: bool = True,
    fixed_range: tuple[float, float] = (0.0, 0.35),
    gamma: float = 1.0,
) -> np.ndarray:
    """Normalize a 2D float channel to ``[0, 1]`` cleanly ignoring NaNs."""
    data = np.asarray(arr, dtype=np.float32)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    if use_percentile_stretch:
        lo = float(np.percentile(valid, 2.0))
        hi = float(np.percentile(valid, 98.0))
        if hi <= lo + 1e-6:
            lo = float(np.min(valid))
            hi = float(np.max(valid))
    else:
        lo, hi = float(fixed_range[0]), float(fixed_range[1])

    if hi <= lo + 1e-6:
        hi = lo + 1.0

    norm = np.clip((np.nan_to_num(data, nan=lo) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    g = float(max(gamma, 0.05))
    if abs(g - 1.0) > 1e-3:
        norm = np.power(norm, 1.0 / g)
    return norm


def _build_rgb_rgba_array(
    raster: xr.Dataset | xr.DataArray,
    bands: Sequence[str] | None,
    *,
    use_percentile_stretch: bool = True,
    gamma: float = 1.0,
    fixed_ranges: Sequence[tuple[float, float]] | None = None,
    y_ascending: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build a north-up RGBA uint8 array (H, W, 4) for 3-band optical or 2-band SAR."""
    if isinstance(raster, xr.Dataset):
        req_bands = list(bands) if bands else list(raster.data_vars)[:3]
        if len(req_bands) == 2 and set(req_bands) == {"VV", "VH"}:
            vv = np.asarray(raster["VV"].values, dtype=np.float32)
            vh = np.asarray(raster["VH"].values, dtype=np.float32)
            ch_arrays = [vv, vh, vv - vh]
            legend_bands = ["VV", "VH", "VV-VH"]
            if fixed_ranges is None:
                fixed_ranges = [(-22.0, -2.0), (-28.0, -8.0), (2.0, 14.0)]
        elif len(req_bands) >= 3:
            ch_arrays = [np.asarray(raster[b].values, dtype=np.float32) for b in req_bands[:3]]
            legend_bands = list(req_bands[:3])
        else:
            raise ValueError(f"RGB rendering requires 3 bands (or SAR VV+VH), got {req_bands!r}.")
    elif isinstance(raster, xr.DataArray) and raster.ndim == 3:
        arr3 = np.asarray(raster.values, dtype=np.float32)
        ch_arrays = [arr3[0], arr3[1], arr3[2]]
        legend_bands = list(bands[:3]) if (bands and len(bands) >= 3) else ["R", "G", "B"]
    else:
        raise TypeError("RGB mode requires an xarray.Dataset or 3D xarray.DataArray.")

    valid = np.isfinite(ch_arrays[0]) & np.isfinite(ch_arrays[1]) & np.isfinite(ch_arrays[2])
    h, w = ch_arrays[0].shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for idx in range(3):
        f_rng = (
            fixed_ranges[idx]
            if (fixed_ranges is not None and idx < len(fixed_ranges))
            else (0.0, 0.35)
        )
        ch01 = _stretch_channel_01(
            ch_arrays[idx],
            use_percentile_stretch=use_percentile_stretch,
            fixed_range=f_rng,
            gamma=gamma,
        )
        rgba[..., idx] = np.where(valid, np.clip(np.round(ch01 * 255.0), 0, 255), 238).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)

    if y_ascending:
        rgba = np.flipud(rgba)
    return rgba, legend_bands


def _project_aoi_to_crs(
    dst_crs_spec: int | str | CRS,
    aoi_polygon_geojson: dict[str, Any] | None = None,
    aoi_bbox_4326: Sequence[float] | None = None,
) -> Any | None:
    """Reproject an EPSG:4326 GeoJSON polygon or bbox into the raster's metric CRS."""
    geom_4326 = None
    if aoi_polygon_geojson is not None:
        try:
            geom_4326 = shape(aoi_polygon_geojson)
        except Exception:
            geom_4326 = None
    elif aoi_bbox_4326 is not None and len(aoi_bbox_4326) == 4:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in aoi_bbox_4326)
        rect = box(min_lon, min_lat, max_lon, max_lat)
        max_seg = max((max_lon - min_lon) / 16.0, (max_lat - min_lat) / 16.0, 0.0005)
        geom_4326 = rect.segmentize(max_segment_length=max_seg)

    if geom_4326 is None or geom_4326.is_empty:
        return None

    dst_crs = normalize_crs(dst_crs_spec)
    transformer = Transformer.from_crs(CRS.from_user_input(CRS_WGS84), dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom_4326)


# ---------------------------------------------------------------------------
# Cartographic Drawing Helpers (Scale Bar, North Arrow, AOI Outline)
# ---------------------------------------------------------------------------
def _draw_aoi_outline(ax: plt.Axes, geom_metric: Any) -> None:
    """Plot a high-contrast cartographic AOI boundary (white halo + teal dashed line)."""
    if geom_metric is None or geom_metric.is_empty:
        return

    geoms = list(geom_metric.geoms) if hasattr(geom_metric, "geoms") else [geom_metric]
    for poly in geoms:
        if hasattr(poly, "exterior"):
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color="#FFFFFF", linewidth=2.8, alpha=0.88, zorder=7)
            ax.plot(xs, ys, color=COLOR_TEAL, linewidth=1.6, linestyle="--", zorder=8)
            for interior in poly.interiors:
                ixs, iys = interior.xy
                ax.plot(ixs, iys, color="#FFFFFF", linewidth=2.2, alpha=0.85, zorder=7)
                ax.plot(ixs, iys, color=COLOR_TEAL, linewidth=1.2, linestyle="--", zorder=8)


def _draw_scale_bar(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    scalebar_m: int,
    scale_denom: int,
) -> None:
    """Draw a classic alternating B/W geodetic scale bar + numerical scale in bottom-left."""
    min_e, max_e, min_n, max_n = extent
    width_m = max_e - min_e
    height_m = max_n - min_n

    # Anchor box in data coordinates (bottom-left with clean margin)
    margin_x = width_m * 0.025
    margin_y = height_m * 0.03
    pad_x = width_m * 0.018
    pad_y_bot = height_m * 0.036
    pad_y_top = height_m * 0.052
    bar_h = height_m * 0.016

    x0 = min_e + margin_x + pad_x
    y0 = min_n + margin_y + pad_y_bot

    box_x = x0 - pad_x
    box_y = min_n + margin_y
    box_w = float(scalebar_m) + 2.0 * pad_x
    box_h = pad_y_bot + bar_h + pad_y_top

    bg = mpatches.FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        boxstyle=f"round,pad=0,rounding_size={min(width_m, height_m) * 0.008}",
        facecolor="#FFFFFF",
        edgecolor=COLOR_SLATE_700,
        linewidth=0.75,
        alpha=0.90,
        zorder=10,
    )
    ax.add_patch(bg)

    # 4 alternating geodetic segments (black / white / black / white)
    n_seg = 4
    seg_w = float(scalebar_m) / float(n_seg)
    for i in range(n_seg):
        seg_rect = mpatches.Rectangle(
            (x0 + i * seg_w, y0),
            seg_w,
            bar_h,
            facecolor=COLOR_SLATE_900 if (i % 2 == 0) else "#FFFFFF",
            edgecolor=COLOR_SLATE_900,
            linewidth=0.75,
            zorder=11,
        )
        ax.add_patch(seg_rect)

    # Tick labels below scale bar (0, half, full)
    label_y = y0 - height_m * 0.008
    ax.text(
        x0,
        label_y,
        "0",
        fontsize=7.2,
        fontweight="bold",
        color=COLOR_SLATE_900,
        ha="center",
        va="top",
        zorder=12,
    )
    ax.text(
        x0 + 0.5 * scalebar_m,
        label_y,
        _format_scalebar_label(0.5 * scalebar_m),
        fontsize=7.2,
        fontweight="bold",
        color=COLOR_SLATE_900,
        ha="center",
        va="top",
        zorder=12,
    )
    ax.text(
        x0 + scalebar_m,
        label_y,
        _format_scalebar_label(scalebar_m),
        fontsize=7.2,
        fontweight="bold",
        color=COLOR_SLATE_900,
        ha="center",
        va="top",
        zorder=12,
    )

    # Approximate printed numerical scale right above the bar
    ax.text(
        x0 + 0.5 * scalebar_m,
        y0 + bar_h + height_m * 0.012,
        f"M ≈ 1 : {format_metric_number(scale_denom)}",
        fontsize=7.8,
        fontweight="bold",
        color=COLOR_SLATE_900,
        ha="center",
        va="bottom",
        zorder=12,
    )


def _draw_north_arrow(ax: plt.Axes, lang: str = "sl") -> None:
    """Draw a minimalist two-tone cartographic North arrow in the top-right corner."""
    # Use axes fraction coordinates so the arrow keeps crisp proportions at any aspect ratio
    trans = ax.transAxes
    fig = ax.figure
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_aspect = max(bbox.width / max(bbox.height, 1.0), 0.35)

    # Compact square-ish frame in top-right (in axes fraction coords)
    box_h = 0.135
    box_w = box_h / ax_aspect * 0.78
    margin_y = 0.025
    margin_x = margin_y / ax_aspect
    bx = 1.0 - margin_x - box_w
    by = 1.0 - margin_y - box_h

    frame = mpatches.FancyBboxPatch(
        (bx, by),
        box_w,
        box_h,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=trans,
        facecolor="#FFFFFF",
        edgecolor=COLOR_SLATE_700,
        linewidth=0.75,
        alpha=0.90,
        zorder=10,
    )
    ax.add_patch(frame)

    cx = bx + 0.5 * box_w
    tip_y = by + box_h * 0.71
    base_y = by + box_h * 0.14
    notch_y = by + box_h * 0.27
    wing_dx = box_w * 0.27

    # Left half (solid dark slate)
    left_poly = mpatches.Polygon(
        [(cx, tip_y), (cx - wing_dx, base_y), (cx, notch_y)],
        closed=True,
        transform=trans,
        facecolor=COLOR_SLATE_900,
        edgecolor=COLOR_SLATE_900,
        linewidth=0.75,
        zorder=11,
    )
    # Right half (crisp white with dark stroke)
    right_poly = mpatches.Polygon(
        [(cx, tip_y), (cx + wing_dx, base_y), (cx, notch_y)],
        closed=True,
        transform=trans,
        facecolor="#FFFFFF",
        edgecolor=COLOR_SLATE_900,
        linewidth=0.75,
        zorder=11,
    )
    ax.add_patch(left_poly)
    ax.add_patch(right_poly)

    north_letter = "S" if lang == "sl" else "N"
    ax.text(
        cx,
        by + box_h * 0.86,
        north_letter,
        transform=trans,
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_SLATE_900,
        ha="center",
        va="center",
        zorder=12,
    )


# ---------------------------------------------------------------------------
# Main Geodetic Print Layout Renderer
# ---------------------------------------------------------------------------
def render_geodetic_map(
    raster: xr.Dataset | xr.DataArray,
    crs: int | str | CRS = CRS_D96TM,
    *,
    is_rgb: bool = True,
    bands: Sequence[str] | None = None,
    use_percentile_stretch: bool = True,
    gamma: float = 1.0,
    fixed_ranges: Sequence[tuple[float, float]] | None = None,
    cmap_name: str = "RdYlGn",
    vmin: float = -0.2,
    vmax: float = 0.9,
    discrete_items: Sequence[tuple[int, str, str]] | None = None,
    legend_title: str = "",
    legend_units: str = "",
    stats_summary: Sequence[tuple[str, str]] | None = None,
    title: str = "Satelitska karta območja",
    subtitle: str = "",
    author: str = "Oddelek za geodezijo",
    sensor_label: str = "Sentinel-2 L2A",
    date_label: str = "",
    resolution_m: float = 10.0,
    data_source: str = "Copernicus / USGS (Planetary Computer STAC)",
    page_format: str = "a4_landscape",
    show_grid: bool = True,
    grid_spacing_m: int | None = None,
    grid_style: str = "light",
    show_scalebar: bool = True,
    show_north_arrow: bool = True,
    show_legend: bool = True,
    show_aoi_outline: bool = True,
    aoi_polygon_geojson: dict[str, Any] | None = None,
    aoi_bbox_4326: Sequence[float] | None = None,
    aoi_legend_label: str | None = None,
    lang: str = "sl",
    preview_dpi: int = 150,
    export_dpi: int = 300,
) -> dict[str, Any]:
    """Render a publication-grade geodetic map in strict metric projection (PNG + PDF).

    Parameters
    ----------
    raster : xr.Dataset | xr.DataArray
        Georeferenced raster in a projected metric CRS (``EPSG:3794`` or ``EPSG:32633``).
    crs : int | str | CRS
        Projected CRS matching ``raster`` (e.g. ``"EPSG:3794"``).
    is_rgb : bool
        ``True`` for 3-band optical or 2-band SAR false-color composites; ``False`` for
        single-band continuous indices / SAR dB or discrete masks.
    discrete_items : Sequence[tuple[int, str, str]] | None
        Optional discrete class definitions ``(class_value, label, hex_color)`` for
        classified rasters (e.g., SAR Water / Flood Mask).
    page_format : str
        One of ``"a4_landscape"``, ``"a4_portrait"``, ``"square"``.
    grid_spacing_m : int | None
        Manual metric grid spacing in meters, or ``None`` for automatic calculation.
    grid_style : str
        ``"light"`` (semi-transparent white gridlines) or ``"dark"`` (dark slate gridlines).

    Returns
    -------
    dict[str, Any]
        Contains ``png_preview_bytes``, ``png_hires_bytes``, ``pdf_bytes``,
        ``scale_denom``, ``grid_spacing_m``, ``scalebar_length_m``, and ``extent_m``.
    """
    fmt_spec = PAGE_FORMATS.get(page_format, PAGE_FORMATS["a4_landscape"])
    fig_w_in, fig_h_in = fmt_spec["figsize"]
    orientation = fmt_spec["orientation"]

    min_e, max_e, min_n, max_n, y_ascending = _extract_extent_and_orientation(raster)
    extent = (min_e, max_e, min_n, max_n)
    width_m = max(max_e - min_e, 1.0)
    height_m = max(max_n - min_n, 1.0)

    crs_title, x_lbl, y_lbl = _crs_display_info(crs, lang=lang)
    eff_grid_m = (
        int(grid_spacing_m)
        if (grid_spacing_m is not None and int(grid_spacing_m) > 0)
        else compute_nice_grid_spacing(max(width_m, height_m), target_lines=5)
    )
    scalebar_m = compute_nice_scalebar_length(width_m, target_fraction=0.22)

    aoi_metric = (
        _project_aoi_to_crs(crs, aoi_polygon_geojson=aoi_polygon_geojson, aoi_bbox_4326=aoi_bbox_4326)
        if show_aoi_outline
        else None
    )

    # Prepare raster display arrays
    rgb_rgba: np.ndarray | None = None
    legend_bands: list[str] = []
    scalar_arr: np.ndarray | None = None
    discrete_rgba: np.ndarray | None = None

    if is_rgb:
        rgb_rgba, legend_bands = _build_rgb_rgba_array(
            raster,
            bands=bands,
            use_percentile_stretch=use_percentile_stretch,
            gamma=gamma,
            fixed_ranges=fixed_ranges,
            y_ascending=y_ascending,
        )
    else:
        if isinstance(raster, xr.Dataset):
            var_name = bands[0] if bands else next(iter(raster.data_vars))
            da = raster[var_name]
        else:
            da = raster
        vals = np.asarray(da.values, dtype=np.float32)
        if y_ascending:
            vals = np.flipud(vals)

        if discrete_items:
            h, w = vals.shape
            discrete_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            for cls_val, _lbl, hex_col in discrete_items:
                r_f, g_f, b_f, a_f = mcolors.to_rgba(hex_col)
                mask = np.isfinite(vals) & (np.round(vals).astype(np.int32) == int(cls_val))
                discrete_rgba[mask] = (
                    int(round(r_f * 255)),
                    int(round(g_f * 255)),
                    int(round(b_f * 255)),
                    int(round(a_f * 255)),
                )
        else:
            scalar_arr = vals

    # -----------------------------------------------------------------------
    # Build Matplotlib Figure with Clean Cartographic Layout Regions
    # -----------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": COLOR_SLATE_900,
            "axes.linewidth": 1.0,
        }
    )

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor="#FFFFFF")
    try:
        # Outer neatline border around the entire cartographic sheet
        sheet_border = mpatches.Rectangle(
            (0.015, 0.015),
            0.970,
            0.970,
            transform=fig.transFigure,
            facecolor="#FFFFFF",
            edgecolor=COLOR_SLATE_700,
            linewidth=1.1,
            zorder=0,
        )
        fig.patches.append(sheet_border)

        # Header accent bar (crisp cartographic teal)
        header_top = 0.968
        header_h = 0.088 if orientation == "landscape" else 0.072
        header_y = header_top - header_h

        accent_bar = mpatches.Rectangle(
            (0.025, header_y + 0.008),
            0.006,
            header_h - 0.014,
            transform=fig.transFigure,
            facecolor=COLOR_TEAL,
            edgecolor="none",
            zorder=2,
        )
        fig.patches.append(accent_bar)

        # Header divider line
        fig.lines.append(
            plt.Line2D(
                [0.025, 0.975],
                [header_y, header_y],
                transform=fig.transFigure,
                color=COLOR_SLATE_300,
                linewidth=0.9,
            )
        )

        # Title, Subtitle, Author
        title_clean = (title or "").strip() or (
            "Satelitska karta območja" if lang == "sl" else "Satellite Map of Area of Interest"
        )
        fig.text(
            0.039,
            header_y + header_h * 0.62,
            title_clean,
            fontsize=14.5 if orientation == "landscape" else 13.5,
            fontweight="bold",
            color=COLOR_SLATE_900,
            ha="left",
            va="center",
        )
        if subtitle and subtitle.strip():
            fig.text(
                0.039,
                header_y + header_h * 0.24,
                subtitle.strip(),
                fontsize=9.2,
                color=COLOR_SLATE_700,
                ha="left",
                va="center",
            )
        if author and author.strip():
            author_prefix = "Avtor:" if lang == "sl" else "Author:"
            fig.text(
                0.968,
                header_y + header_h * 0.60,
                f"{author_prefix} {author.strip()}",
                fontsize=8.6,
                fontweight="bold",
                color=COLOR_SLATE_700,
                ha="right",
                va="center",
            )
        if date_label:
            fig.text(
                0.968,
                header_y + header_h * 0.24,
                f"{sensor_label}  ·  {date_label}",
                fontsize=8.2,
                color=COLOR_SLATE_500,
                ha="right",
                va="center",
            )

        # Footer geometry
        footer_y = 0.022
        footer_h = 0.068 if orientation == "landscape" else 0.060
        footer_top = footer_y + footer_h
        fig.patches.append(
            mpatches.Rectangle(
                (0.025, footer_y),
                0.950,
                footer_h,
                transform=fig.transFigure,
                facecolor=COLOR_BG_LIGHT,
                edgecolor=COLOR_SLATE_300,
                linewidth=0.75,
                zorder=1,
            )
        )

        # Compute map axes and legend panel positions cleanly
        if orientation == "landscape":
            if show_legend:
                map_rect = [0.088, footer_top + 0.058, 0.635, header_y - footer_top - 0.105]
                leg_rect = [0.768, footer_top + 0.032, 0.202, header_y - footer_top - 0.062]
            else:
                map_rect = [0.088, footer_top + 0.058, 0.855, header_y - footer_top - 0.105]
                leg_rect = None
        else:
            if show_legend:
                leg_h = 0.125 if orientation == "portrait" else 0.145
                leg_rect = [0.035, footer_top + 0.014, 0.930, leg_h]
                map_bot = footer_top + leg_h + 0.065
                map_rect = [0.105, map_bot, 0.825, header_y - map_bot - 0.045]
            else:
                map_rect = [0.105, footer_top + 0.060, 0.825, header_y - footer_top - 0.105]
                leg_rect = None

        ax = fig.add_axes(map_rect)
        ax.set_facecolor(COLOR_NODATA)

        # Render Raster strictly in projected metric coordinates (1:1 equal aspect)
        mappable = None
        if is_rgb and rgb_rgba is not None:
            ax.imshow(
                rgb_rgba,
                extent=extent,
                origin="upper",
                interpolation="nearest",
                zorder=2,
            )
        elif discrete_rgba is not None:
            ax.imshow(
                discrete_rgba,
                extent=extent,
                origin="upper",
                interpolation="nearest",
                zorder=2,
            )
        elif scalar_arr is not None:
            cmap = matplotlib.colormaps[cmap_name].copy()
            cmap.set_bad(color=COLOR_NODATA)
            mappable = ax.imshow(
                scalar_arr,
                extent=extent,
                origin="upper",
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                interpolation="nearest",
                zorder=2,
            )

        ax.set_xlim(min_e, max_e)
        ax.set_ylim(min_n, max_n)
        ax.set_aspect("equal", adjustable="box")

        # Draw exact AOI polygon outline in metric CRS
        if aoi_metric is not None:
            _draw_aoi_outline(ax, aoi_metric)

        # Coordinate Grid & Metric Tick Labels
        x_ticks = np.arange(
            math.ceil(min_e / eff_grid_m) * eff_grid_m,
            max_e + 1e-6,
            eff_grid_m,
            dtype=np.float64,
        )
        y_ticks = np.arange(
            math.ceil(min_n / eff_grid_m) * eff_grid_m,
            max_n + 1e-6,
            eff_grid_m,
            dtype=np.float64,
        )
        # Fallback if raster span is smaller than chosen manual grid step
        if x_ticks.size < 2:
            x_ticks = np.linspace(min_e, max_e, 3)
        if y_ticks.size < 2:
            y_ticks = np.linspace(min_n, max_n, 3)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        metric_fmt = mticker.FuncFormatter(lambda val, _pos: format_metric_number(val))
        ax.xaxis.set_major_formatter(metric_fmt)
        ax.yaxis.set_major_formatter(metric_fmt)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4.0,
            width=0.8,
            labelsize=7.6,
            colors=COLOR_SLATE_900,
            top=True,
            right=True,
            labeltop=True,
            labelright=False,
        )
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90)
            lbl.set_va("center")

        ax.set_xlabel(x_lbl, fontsize=8.2, fontweight="bold", color=COLOR_SLATE_900, labelpad=4)
        ax.set_ylabel(y_lbl, fontsize=8.2, fontweight="bold", color=COLOR_SLATE_900, labelpad=4)

        if show_grid:
            if grid_style == "dark":
                g_color, g_alpha = COLOR_SLATE_900, 0.42
            else:
                g_color, g_alpha = "#FFFFFF", 0.62
            ax.grid(
                True,
                which="major",
                color=g_color,
                linestyle="-",
                linewidth=0.65,
                alpha=g_alpha,
                zorder=5,
            )

        # Force layout draw so we get the EXACT rendered map width on paper for true numerical scale!
        fig.canvas.draw()
        ax_bbox_in = ax.get_position()
        paper_width_m = float(ax_bbox_in.width * fig_w_in) * 0.0254
        scale_denom = compute_nice_scale_denom(width_m, paper_width_m)

        # Scale bar & North arrow
        if show_scalebar:
            _draw_scale_bar(ax, extent=extent, scalebar_m=scalebar_m, scale_denom=scale_denom)
        if show_north_arrow:
            _draw_north_arrow(ax, lang=lang)

        # -------------------------------------------------------------------
        # Dedicated Cartographic Legend Panel (never overlaps raster)
        # -------------------------------------------------------------------
        if show_legend and leg_rect is not None:
            lx, ly, lw, lh = leg_rect
            leg_frame = mpatches.FancyBboxPatch(
                (lx, ly),
                lw,
                lh,
                boxstyle="round,pad=0.0,rounding_size=0.006",
                transform=fig.transFigure,
                facecolor="#FFFFFF",
                edgecolor=COLOR_SLATE_300,
                linewidth=0.85,
                zorder=2,
            )
            fig.patches.append(leg_frame)

            leg_heading = "LEGENDA" if lang == "sl" else "LEGEND"
            fig.text(
                lx + 0.012,
                ly + lh - 0.020,
                leg_heading,
                fontsize=8.6,
                fontweight="bold",
                color=COLOR_TEAL,
                ha="left",
                va="top",
            )
            fig.lines.append(
                plt.Line2D(
                    [lx + 0.010, lx + lw - 0.010],
                    [ly + lh - 0.040, ly + lh - 0.040],
                    transform=fig.transFigure,
                    color=COLOR_SLATE_300,
                    linewidth=0.7,
                )
            )

            if is_rgb:
                # Compact Band Combination Legend
                subtitle_lbl = legend_title or (
                    "Kombinacija spektralnih pasov" if lang == "sl" else "Spectral Band Combination"
                )
                fig.text(
                    lx + 0.012,
                    ly + lh - 0.056,
                    subtitle_lbl,
                    fontsize=8.3,
                    fontweight="bold",
                    color=COLOR_SLATE_900,
                    ha="left",
                    va="top",
                )
                ch_colors = ("#DC2626", "#16A34A", "#2563EB")
                ch_prefixes = ("R", "G", "B")
                if orientation == "landscape":
                    cur_y = ly + lh - 0.100
                    for idx, b_code in enumerate(legend_bands[:3]):
                        b_desc = BAND_WAVELENGTH_LABELS.get(b_code, b_code)
                        fig.patches.append(
                            mpatches.Rectangle(
                                (lx + 0.014, cur_y - 0.012),
                                0.014,
                                0.018,
                                transform=fig.transFigure,
                                facecolor=ch_colors[idx],
                                edgecolor=COLOR_SLATE_700,
                                linewidth=0.5,
                                zorder=3,
                            )
                        )
                        fig.text(
                            lx + 0.034,
                            cur_y - 0.003,
                            f"{ch_prefixes[idx]}: {b_desc}",
                            fontsize=7.8,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )
                        cur_y -= 0.038
                    if aoi_metric is not None:
                        cur_y -= 0.012
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.014, lx + 0.032],
                                [cur_y, cur_y],
                                transform=fig.transFigure,
                                color=COLOR_TEAL,
                                linestyle="--",
                                linewidth=1.6,
                            )
                        )
                        fig.text(
                            lx + 0.038,
                            cur_y,
                            aoi_legend_label or ("Meja AOI" if lang == "sl" else "AOI Boundary"),
                            fontsize=7.8,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )
                    if stats_summary:
                        cur_y -= 0.042
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.010, lx + lw - 0.010],
                                [cur_y + 0.018, cur_y + 0.018],
                                transform=fig.transFigure,
                                color=COLOR_SLATE_300,
                                linewidth=0.6,
                            )
                        )
                        for k_lbl, v_lbl in stats_summary[:5]:
                            fig.text(
                                lx + 0.014,
                                cur_y,
                                f"{k_lbl}:",
                                fontsize=7.5,
                                color=COLOR_SLATE_500,
                                ha="left",
                                va="center",
                            )
                            fig.text(
                                lx + lw - 0.012,
                                cur_y,
                                v_lbl,
                                fontsize=7.6,
                                fontweight="bold",
                                color=COLOR_SLATE_900,
                                ha="right",
                                va="center",
                            )
                            cur_y -= 0.030
                else:
                    # Horizontal compact layout for portrait / square
                    step_x = (lw - 0.030) / 3.0
                    row_y = ly + lh * 0.34
                    for idx, b_code in enumerate(legend_bands[:3]):
                        bx_i = lx + 0.016 + idx * step_x
                        b_desc = BAND_WAVELENGTH_LABELS.get(b_code, b_code)
                        fig.patches.append(
                            mpatches.Rectangle(
                                (bx_i, row_y - 0.009),
                                0.018,
                                0.016,
                                transform=fig.transFigure,
                                facecolor=ch_colors[idx],
                                edgecolor=COLOR_SLATE_700,
                                linewidth=0.5,
                                zorder=3,
                            )
                        )
                        fig.text(
                            bx_i + 0.024,
                            row_y - 0.001,
                            f"{ch_prefixes[idx]}: {b_desc}",
                            fontsize=7.6,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )
                    if aoi_metric is not None:
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.016, lx + 0.044],
                                [ly + lh * 0.14, ly + lh * 0.14],
                                transform=fig.transFigure,
                                color=COLOR_TEAL,
                                linestyle="--",
                                linewidth=1.6,
                            )
                        )
                        fig.text(
                            lx + 0.052,
                            ly + lh * 0.14,
                            aoi_legend_label or ("Meja AOI" if lang == "sl" else "AOI Boundary"),
                            fontsize=7.6,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )

            elif discrete_items:
                # Discrete class legend (SAR Water Mask / Land Cover)
                title_str = legend_title or ("Razredi" if lang == "sl" else "Classes")
                fig.text(
                    lx + 0.012,
                    ly + lh - 0.056,
                    title_str,
                    fontsize=8.3,
                    fontweight="bold",
                    color=COLOR_SLATE_900,
                    ha="left",
                    va="top",
                )
                if orientation == "landscape":
                    cur_y = ly + lh - 0.100
                    for _val, cls_lbl, hex_col in discrete_items:
                        fig.patches.append(
                            mpatches.Rectangle(
                                (lx + 0.014, cur_y - 0.012),
                                0.016,
                                0.018,
                                transform=fig.transFigure,
                                facecolor=hex_col,
                                edgecolor=COLOR_SLATE_700,
                                linewidth=0.55,
                                zorder=3,
                            )
                        )
                        fig.text(
                            lx + 0.036,
                            cur_y - 0.003,
                            cls_lbl,
                            fontsize=7.9,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )
                        cur_y -= 0.038
                    if aoi_metric is not None:
                        cur_y -= 0.010
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.014, lx + 0.032],
                                [cur_y, cur_y],
                                transform=fig.transFigure,
                                color=COLOR_TEAL,
                                linestyle="--",
                                linewidth=1.6,
                            )
                        )
                        fig.text(
                            lx + 0.038,
                            cur_y,
                            aoi_legend_label or ("Meja AOI" if lang == "sl" else "AOI Boundary"),
                            fontsize=7.8,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )
                    if stats_summary:
                        cur_y -= 0.042
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.010, lx + lw - 0.010],
                                [cur_y + 0.018, cur_y + 0.018],
                                transform=fig.transFigure,
                                color=COLOR_SLATE_300,
                                linewidth=0.6,
                            )
                        )
                        for k_lbl, v_lbl in stats_summary[:5]:
                            fig.text(
                                lx + 0.014,
                                cur_y,
                                f"{k_lbl}:",
                                fontsize=7.5,
                                color=COLOR_SLATE_500,
                                ha="left",
                                va="center",
                            )
                            fig.text(
                                lx + lw - 0.012,
                                cur_y,
                                v_lbl,
                                fontsize=7.6,
                                fontweight="bold",
                                color=COLOR_SLATE_900,
                                ha="right",
                                va="center",
                            )
                            cur_y -= 0.030
                else:
                    n_cls = max(len(discrete_items), 1)
                    step_x = (lw - 0.030) / float(min(n_cls, 4))
                    row_y = ly + lh * 0.34
                    for idx, (_val, cls_lbl, hex_col) in enumerate(discrete_items[:4]):
                        bx_i = lx + 0.016 + idx * step_x
                        fig.patches.append(
                            mpatches.Rectangle(
                                (bx_i, row_y - 0.009),
                                0.018,
                                0.016,
                                transform=fig.transFigure,
                                facecolor=hex_col,
                                edgecolor=COLOR_SLATE_700,
                                linewidth=0.55,
                                zorder=3,
                            )
                        )
                        fig.text(
                            bx_i + 0.024,
                            row_y - 0.001,
                            cls_lbl,
                            fontsize=7.8,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )

            elif mappable is not None:
                # Continuous Colorbar Legend (Spectral Index / SAR dB / Change Detection)
                unit_suffix = f" [{legend_units}]" if legend_units else ""
                cbar_title = f"{legend_title}{unit_suffix}" if legend_title else (
                    "Vrednost" if lang == "sl" else "Value"
                )
                fig.text(
                    lx + 0.012,
                    ly + lh - 0.056,
                    cbar_title,
                    fontsize=8.3,
                    fontweight="bold",
                    color=COLOR_SLATE_900,
                    ha="left",
                    va="top",
                )
                ticks = np.linspace(float(vmin), float(vmax), 5)
                tick_fmt = "{:.1f}" if abs(vmax - vmin) >= 5.0 else "{:.2f}"

                if orientation == "landscape":
                    cbar_h = min(lh * 0.44, 0.34)
                    cbar_y = ly + lh - 0.095 - cbar_h
                    cax = fig.add_axes([lx + 0.022, cbar_y, 0.024, cbar_h])
                    cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
                    cb.set_ticks(ticks)
                    cb.set_ticklabels([tick_fmt.format(t_v) for t_v in ticks])
                    cb.ax.tick_params(labelsize=7.6, colors=COLOR_SLATE_900, length=3.0)
                    cb.outline.set_edgecolor(COLOR_SLATE_700)
                    cb.outline.set_linewidth(0.65)

                    cur_y = cbar_y - 0.036
                    if aoi_metric is not None:
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.014, lx + 0.032],
                                [cur_y, cur_y],
                                transform=fig.transFigure,
                                color=COLOR_TEAL,
                                linestyle="--",
                                linewidth=1.6,
                            )
                        )
                        fig.text(
                            lx + 0.038,
                            cur_y,
                            aoi_legend_label or ("Meja AOI" if lang == "sl" else "AOI Boundary"),
                            fontsize=7.8,
                            color=COLOR_SLATE_900,
                            ha="left",
                            va="center",
                        )
                        cur_y -= 0.036

                    if stats_summary:
                        fig.lines.append(
                            plt.Line2D(
                                [lx + 0.010, lx + lw - 0.010],
                                [cur_y + 0.016, cur_y + 0.016],
                                transform=fig.transFigure,
                                color=COLOR_SLATE_300,
                                linewidth=0.6,
                            )
                        )
                        for k_lbl, v_lbl in stats_summary[:5]:
                            if cur_y < ly + 0.014:
                                break
                            fig.text(
                                lx + 0.014,
                                cur_y,
                                f"{k_lbl}:",
                                fontsize=7.4,
                                color=COLOR_SLATE_500,
                                ha="left",
                                va="center",
                            )
                            fig.text(
                                lx + lw - 0.012,
                                cur_y,
                                v_lbl,
                                fontsize=7.5,
                                fontweight="bold",
                                color=COLOR_SLATE_900,
                                ha="right",
                                va="center",
                            )
                            cur_y -= 0.028
                else:
                    # Horizontal colorbar for portrait / square
                    cax = fig.add_axes([lx + 0.030, ly + lh * 0.28, lw - 0.060, lh * 0.20])
                    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
                    cb.set_ticks(ticks)
                    cb.set_ticklabels([tick_fmt.format(t_v) for t_v in ticks])
                    cb.ax.tick_params(labelsize=7.5, colors=COLOR_SLATE_900, length=3.0)
                    cb.outline.set_edgecolor(COLOR_SLATE_700)
                    cb.outline.set_linewidth(0.65)

        # -------------------------------------------------------------------
        # Geodetic Technical Metadata Footer
        # -------------------------------------------------------------------
        today_str = date.today().isoformat()
        if lang == "sl":
            row1_left = f"Koordinatni sistem: {crs_title}  |  Merilo: M ≈ 1 : {format_metric_number(scale_denom)}"
            row1_right = f"Prostorska ločljivost: {resolution_m:.0f} m  |  Mreža: {format_metric_number(eff_grid_m)} m"
            row2_left = f"Senzor in datum: {sensor_label}" + (f" ({date_label})" if date_label else "")
            row2_right = f"Vir podatkov: {data_source}  |  Izdelano: {today_str}"
        else:
            row1_left = f"Coordinate System: {crs_title}  |  Scale: M ≈ 1 : {format_metric_number(scale_denom)}"
            row1_right = f"Spatial Resolution: {resolution_m:.0f} m  |  Grid: {format_metric_number(eff_grid_m)} m"
            row2_left = f"Sensor & Date: {sensor_label}" + (f" ({date_label})" if date_label else "")
            row2_right = f"Data Source: {data_source}  |  Created: {today_str}"

        fig.text(
            0.036,
            footer_y + footer_h * 0.68,
            row1_left,
            fontsize=7.6,
            fontweight="bold",
            color=COLOR_SLATE_900,
            ha="left",
            va="center",
        )
        fig.text(
            0.964,
            footer_y + footer_h * 0.68,
            row1_right,
            fontsize=7.5,
            color=COLOR_SLATE_700,
            ha="right",
            va="center",
        )
        fig.text(
            0.036,
            footer_y + footer_h * 0.28,
            row2_left,
            fontsize=7.3,
            color=COLOR_SLATE_700,
            ha="left",
            va="center",
        )
        fig.text(
            0.964,
            footer_y + footer_h * 0.28,
            row2_right,
            fontsize=7.1,
            color=COLOR_SLATE_500,
            ha="right",
            va="center",
        )

        # -------------------------------------------------------------------
        # Export Buffers (150 DPI Preview PNG, 300 DPI Print PNG, 300 DPI PDF)
        # -------------------------------------------------------------------
        buf_prev = io.BytesIO()
        fig.savefig(buf_prev, format="png", dpi=int(preview_dpi), facecolor="#FFFFFF")
        png_preview_bytes = buf_prev.getvalue()
        buf_prev.close()

        buf_hi = io.BytesIO()
        fig.savefig(buf_hi, format="png", dpi=int(export_dpi), facecolor="#FFFFFF")
        png_hires_bytes = buf_hi.getvalue()
        buf_hi.close()

        buf_pdf = io.BytesIO()
        fig.savefig(
            buf_pdf,
            format="pdf",
            dpi=int(export_dpi),
            facecolor="#FFFFFF",
            metadata={
                "Title": title_clean,
                "Author": (author or "").strip(),
                "Subject": f"{sensor_label} | {crs_title}",
                "Creator": "Daljinsko zaznavanje — Geodetic Cartography Module",
            },
        )
        pdf_bytes = buf_pdf.getvalue()
        buf_pdf.close()

        return {
            "png_preview_bytes": png_preview_bytes,
            "png_hires_bytes": png_hires_bytes,
            "pdf_bytes": pdf_bytes,
            "scale_denom": int(scale_denom),
            "grid_spacing_m": int(eff_grid_m),
            "scalebar_length_m": int(scalebar_m),
            "extent_m": extent,
        }
    finally:
        plt.close(fig)
        gc.collect()
