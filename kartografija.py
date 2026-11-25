"""Geodetic Cartography & QGIS-Style Print Layout Composer.

Renders publication-grade metric maps strictly in the raster's true projected CRS
(D96/TM ``EPSG:3794`` or global auto-UTM) with 1:1 equal metric aspect ratio
(zero Web Mercator distortion).

Key Capabilities
----------------
- Unified, JSON-serializable ``DEFAULT_LAYOUT_CONFIG`` + ``merge_layout_config(...)``
  shared by Workspace 4 UI and the AI EO Assistant.
- 4 paper formats (``A4_LANDSCAPE``, ``A4_PORTRAIT``, ``A3_LANDSCAPE``, ``SQUARE_1_1``)
  and 3 cartographic themes (``classic_light``, ``minimal_white``, ``dark_technical``).
- Full placement control for Title Block, Custom Note, Coordinate Grid (``lines``,
  ``crosses``, ``ticks_only``), AOI Outline, North Arrow (3 styles × 3 sizes × 4 corners),
  Scale Bar (2 styles × 4 corners + numeric scale), Legend (right panel, bottom panel,
  or 3 inside-map corners), zero-dependency Vector Inset Locator Map (Slovenia outline
  with cities or global ±3° graticule), and optional geodetic point markers.
- Collision-aware corner stacking so inside-map elements never overlap.
- Exports 150 DPI preview PNG, 300 DPI print PNG, and 300 DPI vector-text PDF with
  immediate ``plt.close(fig)`` + ``gc.collect()`` memory cleanup.
"""

from __future__ import annotations

import copy
import gc
import io
import math
from datetime import date
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rioxarray  # noqa: F401 - registers .rio accessor
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform

from geodezija_crs import (
    CRS_D96TM,
    CRS_WGS84,
    EPSG_D96TM,
    is_in_slovenia,
    normalize_crs,
)

# ---------------------------------------------------------------------------
# Constants: Page Formats, Themes, Grid Steps, Optical Wavelengths
# ---------------------------------------------------------------------------
MM_TO_INCH: float = 1.0 / 25.4

PAGE_FORMATS: dict[str, dict[str, Any]] = {
    "A4_LANDSCAPE": {
        "width_mm": 297.0,
        "height_mm": 210.0,
        "figsize": (297.0 * MM_TO_INCH, 210.0 * MM_TO_INCH),
        "orientation": "landscape",
    },
    "A4_PORTRAIT": {
        "width_mm": 210.0,
        "height_mm": 297.0,
        "figsize": (210.0 * MM_TO_INCH, 297.0 * MM_TO_INCH),
        "orientation": "portrait",
    },
    "A3_LANDSCAPE": {
        "width_mm": 420.0,
        "height_mm": 297.0,
        "figsize": (420.0 * MM_TO_INCH, 297.0 * MM_TO_INCH),
        "orientation": "landscape",
    },
    "SQUARE_1_1": {
        "width_mm": 210.0,
        "height_mm": 210.0,
        "figsize": (210.0 * MM_TO_INCH, 210.0 * MM_TO_INCH),
        "orientation": "square",
    },
}

# Legacy lowercase aliases kept for 100% backward compatibility
_FORMAT_ALIASES: dict[str, str] = {
    "a4_landscape": "A4_LANDSCAPE",
    "a4_portrait": "A4_PORTRAIT",
    "a3_landscape": "A3_LANDSCAPE",
    "square": "SQUARE_1_1",
    "square_1_1": "SQUARE_1_1",
}

THEME_STYLES: dict[str, dict[str, Any]] = {
    "classic_light": {
        "paper_bg": "#FFFFFF",
        "panel_bg": "#FFFFFF",
        "footer_bg": "#F8FAFC",
        "nodata_bg": "#EEF2F6",
        "text_primary": "#0F172A",
        "text_secondary": "#334155",
        "text_muted": "#64748B",
        "border": "#334155",
        "divider": "#CBD5E1",
        "accent": "#0F766E",
        "overlay_bg": "#FFFFFF",
        "overlay_edge": "#334155",
        "overlay_alpha": 0.92,
        "sheet_border": True,
        "inset_land": "#CCFBF1",
        "inset_land_edge": "#0F766E",
        "inset_bg": "#F8FAFC",
    },
    "minimal_white": {
        "paper_bg": "#FFFFFF",
        "panel_bg": "#FFFFFF",
        "footer_bg": "#FFFFFF",
        "nodata_bg": "#F1F5F9",
        "text_primary": "#0F172A",
        "text_secondary": "#475569",
        "text_muted": "#64748B",
        "border": "#94A3B8",
        "divider": "#E2E8F0",
        "accent": "#0F766E",
        "overlay_bg": "#FFFFFF",
        "overlay_edge": "#94A3B8",
        "overlay_alpha": 0.90,
        "sheet_border": False,
        "inset_land": "#E2E8F0",
        "inset_land_edge": "#334155",
        "inset_bg": "#FFFFFF",
    },
    "dark_technical": {
        "paper_bg": "#0F172A",
        "panel_bg": "#1E293B",
        "footer_bg": "#1E293B",
        "nodata_bg": "#1E293B",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "text_muted": "#94A3B8",
        "border": "#475569",
        "divider": "#334155",
        "accent": "#14B8A6",
        "overlay_bg": "#0F172A",
        "overlay_edge": "#475569",
        "overlay_alpha": 0.88,
        "sheet_border": True,
        "inset_land": "#134E4A",
        "inset_land_edge": "#2DD4BF",
        "inset_bg": "#0F172A",
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

# Palette (strictly cartographic teal / slate, zero purple/violet)
COLOR_TEAL = "#0F766E"
COLOR_SLATE_900 = "#0F172A"
COLOR_SLATE_700 = "#334155"
COLOR_SLATE_500 = "#64748B"
COLOR_SLATE_300 = "#CBD5E1"
COLOR_BG_LIGHT = "#F8FAFC"
COLOR_NODATA = "#EEF2F6"

# Lightweight simplified Slovenia national boundary in WGS84 (lon, lat) — zero external dependencies
SLOVENIA_OUTLINE_WGS84: tuple[tuple[float, float], ...] = (
    (13.71, 46.52),  # Peč / Tromeja (Rateče)
    (14.08, 46.46),  # Karavanke W
    (14.54, 46.43),  # Jezersko
    (14.86, 46.51),  # Koroška / Mežica
    (15.05, 46.63),  # Dravograd / Radlje
    (15.56, 46.68),  # Šentilj
    (15.98, 46.69),  # Gornja Radgona
    (16.12, 46.87),  # Goričko N (Kuzma)
    (16.36, 46.84),  # Hodoš
    (16.59, 46.48),  # Lendava / Pince
    (16.25, 46.48),  # Ljutomer
    (16.14, 46.39),  # Ormož
    (15.86, 46.28),  # Haloze
    (15.65, 46.15),  # Rogaška / Bizeljsko
    (15.68, 45.88),  # Brežice
    (15.32, 45.73),  # Gorjanci / Metlika
    (15.35, 45.55),  # Črnomelj / Vinica
    (15.05, 45.45),  # Kolpa S
    (14.70, 45.53),  # Osilnica
    (14.43, 45.56),  # Snežnik
    (14.12, 45.47),  # Ilirska Bistrica / Brkini
    (13.86, 45.45),  # Dragonja
    (13.59, 45.52),  # Piran
    (13.73, 45.56),  # Koper
    (13.86, 45.65),  # Kras / Sežana
    (13.64, 45.85),  # Nova Gorica
    (13.50, 46.00),  # Goriška Brda
    (13.42, 46.24),  # Breginj / Kobarid
    (13.55, 46.37),  # Bovec / Kanin
    (13.68, 46.45),  # Mangart
    (13.71, 46.52),  # Close loop
)

SLOVENIA_REF_CITIES: tuple[tuple[str, float, float], ...] = (
    ("LJ", 14.506, 46.057),
    ("MB", 15.646, 46.555),
    ("KP", 13.730, 45.548),
)

# ---------------------------------------------------------------------------
# Unified QGIS-Style MapLayoutConfig (JSON-serializable)
# ---------------------------------------------------------------------------
DEFAULT_LAYOUT_CONFIG: dict[str, Any] = {
    # 1. Page & Theme
    "paper_format": "A4_LANDSCAPE",
    "theme": "classic_light",
    "dpi_export": 300,
    "dpi_preview": 150,
    "map_border_width": 1.2,
    # 2. Title Block & Metadata
    "show_title": True,
    "title": "",
    "subtitle": "",
    "author": "",
    "title_position": "top_left",
    "show_footer": True,
    "custom_note_text": "",
    "custom_note_position": "bottom_right",
    # 3. Coordinate Grid & Ticks
    "show_grid": True,
    "grid_spacing": "auto",
    "grid_style": "lines",
    "grid_color": "light",
    "grid_alpha": 0.45,
    "show_aoi_outline": True,
    "aoi_outline_color": "#0F766E",
    # 4. North Arrow
    "show_north_arrow": True,
    "north_position": "top_right",
    "north_style": "classic_compass",
    "north_size": "medium",
    # 5. Scale Bar
    "show_scale_bar": True,
    "scale_position": "bottom_left",
    "scale_style": "alternating_bar",
    "show_numeric_scale": True,
    # 6. Legend & Colorbar
    "show_legend": True,
    "legend_position": "right_panel",
    "legend_title": "",
    "show_metadata_in_legend": True,
    # 7. Overview / Inset Locator Map
    "show_inset_map": False,
    "inset_position": "right_panel_bottom",
    # 8. Vector / Point Annotations
    "markers": [],
}

LAYOUT_PRESETS: dict[str, dict[str, Any]] = {
    "thesis": {
        "paper_format": "A4_LANDSCAPE",
        "theme": "classic_light",
        "map_border_width": 1.2,
        "show_title": True,
        "title_position": "top_left",
        "show_footer": True,
        "show_grid": True,
        "grid_style": "lines",
        "grid_color": "light",
        "grid_alpha": 0.45,
        "show_aoi_outline": True,
        "aoi_outline_color": "#0F766E",
        "show_north_arrow": True,
        "north_position": "top_right",
        "north_style": "classic_compass",
        "north_size": "medium",
        "show_scale_bar": True,
        "scale_position": "bottom_left",
        "scale_style": "alternating_bar",
        "show_numeric_scale": True,
        "show_legend": True,
        "legend_position": "right_panel",
        "show_metadata_in_legend": True,
        "show_inset_map": True,
        "inset_position": "right_panel_bottom",
    },
    "poster": {
        "paper_format": "A3_LANDSCAPE",
        "theme": "minimal_white",
        "map_border_width": 0.9,
        "show_title": True,
        "title_position": "inside_top_center",
        "show_footer": False,
        "show_grid": True,
        "grid_style": "ticks_only",
        "grid_color": "dark",
        "grid_alpha": 0.35,
        "show_aoi_outline": False,
        "aoi_outline_color": "#FFFFFF",
        "show_north_arrow": True,
        "north_position": "top_right",
        "north_style": "minimal_arrow",
        "north_size": "small",
        "show_scale_bar": True,
        "scale_position": "bottom_left",
        "scale_style": "simple_line",
        "show_numeric_scale": True,
        "show_legend": True,
        "legend_position": "inside_bottom_right",
        "show_metadata_in_legend": False,
        "show_inset_map": False,
        "inset_position": "bottom_right",
    },
    "crisis": {
        "paper_format": "A4_LANDSCAPE",
        "theme": "dark_technical",
        "map_border_width": 1.3,
        "show_title": True,
        "title_position": "top_left",
        "show_footer": True,
        "show_grid": True,
        "grid_style": "crosses",
        "grid_color": "teal",
        "grid_alpha": 0.65,
        "show_aoi_outline": True,
        "aoi_outline_color": "#DC2626",
        "show_north_arrow": True,
        "north_position": "top_right",
        "north_style": "geodetic_star",
        "north_size": "medium",
        "show_scale_bar": True,
        "scale_position": "bottom_left",
        "scale_style": "alternating_bar",
        "show_numeric_scale": True,
        "show_legend": True,
        "legend_position": "right_panel",
        "show_metadata_in_legend": True,
        "show_inset_map": True,
        "inset_position": "right_panel_bottom",
    },
}

_VALID_TITLE_POS = ("top_left", "top_center", "inside_top_left", "inside_top_center", "none")
_VALID_NOTE_POS = ("bottom_left", "bottom_right", "below_legend", "none")
_VALID_GRID_STYLES = ("lines", "crosses", "ticks_only")
_VALID_GRID_COLORS = ("light", "dark", "teal")
_VALID_CORNERS = ("top_right", "top_left", "bottom_right", "bottom_left")
_VALID_NORTH_STYLES = ("classic_compass", "minimal_arrow", "geodetic_star")
_VALID_NORTH_SIZES = ("small", "medium", "large")
_VALID_SCALE_STYLES = ("alternating_bar", "simple_line")
_VALID_LEGEND_POS = (
    "right_panel",
    "bottom_panel",
    "inside_bottom_right",
    "inside_top_right",
    "inside_bottom_left",
)
_VALID_INSET_POS = ("bottom_right", "top_right", "bottom_left", "top_left", "right_panel_bottom")


def merge_layout_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a validated, JSON-serializable ``MapLayoutConfig`` dict merged over defaults."""
    cfg = copy.deepcopy(DEFAULT_LAYOUT_CONFIG)
    if not isinstance(overrides, dict):
        return cfg

    for key, val in overrides.items():
        if val is None and key not in ("grid_spacing",):
            continue
        if key in cfg:
            cfg[key] = copy.deepcopy(val)

    # Normalize paper format (accepts both uppercase & legacy lowercase)
    raw_fmt = str(cfg.get("paper_format", "A4_LANDSCAPE")).strip()
    norm_fmt = _FORMAT_ALIASES.get(raw_fmt.lower(), raw_fmt.upper())
    cfg["paper_format"] = norm_fmt if norm_fmt in PAGE_FORMATS else "A4_LANDSCAPE"

    # Normalize theme
    if cfg.get("theme") not in THEME_STYLES:
        cfg["theme"] = "classic_light"

    # Numeric bounds
    try:
        cfg["dpi_export"] = max(150, min(600, int(cfg.get("dpi_export", 300))))
    except Exception:
        cfg["dpi_export"] = 300
    try:
        cfg["dpi_preview"] = max(72, min(300, int(cfg.get("dpi_preview", 150))))
    except Exception:
        cfg["dpi_preview"] = 150
    try:
        cfg["map_border_width"] = max(0.4, min(3.0, float(cfg.get("map_border_width", 1.2))))
    except Exception:
        cfg["map_border_width"] = 1.2
    try:
        cfg["grid_alpha"] = max(0.10, min(1.0, float(cfg.get("grid_alpha", 0.45))))
    except Exception:
        cfg["grid_alpha"] = 0.45

    # Grid spacing ("auto" or positive int meters)
    gs = cfg.get("grid_spacing", "auto")
    if gs in (None, "", 0, "0", "auto", "AUTO"):
        cfg["grid_spacing"] = "auto"
    else:
        try:
            gs_i = int(gs)
            cfg["grid_spacing"] = gs_i if gs_i > 0 else "auto"
        except Exception:
            cfg["grid_spacing"] = "auto"

    # Enum validations
    if cfg.get("title_position") not in _VALID_TITLE_POS:
        cfg["title_position"] = "top_left"
    if cfg.get("custom_note_position") not in _VALID_NOTE_POS:
        cfg["custom_note_position"] = "bottom_right"
    if cfg.get("grid_style") not in _VALID_GRID_STYLES:
        cfg["grid_style"] = "lines"
    if cfg.get("grid_color") not in _VALID_GRID_COLORS:
        cfg["grid_color"] = "light"
    if cfg.get("north_position") not in _VALID_CORNERS:
        cfg["north_position"] = "top_right"
    if cfg.get("north_style") not in _VALID_NORTH_STYLES:
        cfg["north_style"] = "classic_compass"
    if cfg.get("north_size") not in _VALID_NORTH_SIZES:
        cfg["north_size"] = "medium"
    if cfg.get("scale_position") not in _VALID_CORNERS:
        cfg["scale_position"] = "bottom_left"
    if cfg.get("scale_style") not in _VALID_SCALE_STYLES:
        cfg["scale_style"] = "alternating_bar"
    if cfg.get("legend_position") not in _VALID_LEGEND_POS:
        cfg["legend_position"] = "right_panel"
    if cfg.get("inset_position") not in _VALID_INSET_POS:
        cfg["inset_position"] = "right_panel_bottom"

    if not isinstance(cfg.get("markers"), list):
        cfg["markers"] = []

    return cfg


# ---------------------------------------------------------------------------
# Metric Calculation Helpers
# ---------------------------------------------------------------------------
def format_metric_number(value: float | int) -> str:
    """Format an integer/metric coordinate with space thousands separators."""
    return f"{int(round(value)):,}".replace(",", " ")


def _clean_print_text(text: str | None) -> str:
    """Remove non-BMP emoji glyphs unsupported by DejaVu Sans while keeping diacritics and math symbols."""
    if not text:
        return ""
    cleaned = "".join(ch for ch in str(text) if ord(ch) <= 0xFFFF and ch != "\ufe0f")
    return " ".join(cleaned.split())


def compute_nice_grid_spacing(span_m: float, target_lines: int = 5) -> int:
    """Select a clean round metric grid interval in meters for a given map span."""
    span = max(float(span_m), 100.0)
    raw = span / max(float(target_lines), 2.0)
    for step in NICE_GRID_INTERVALS_M:
        if step >= raw * 0.85:
            return step
    return NICE_GRID_INTERVALS_M[-1]


def compute_nice_scalebar_length(width_m: float, target_fraction: float = 0.22) -> int:
    """Select a clean round metric scale-bar length (~20–25% of map width)."""
    target = max(float(width_m) * float(target_fraction), 50.0)
    best = NICE_SCALEBAR_LENGTHS_M[0]
    best_dist = abs(best - target)
    for cand in NICE_SCALEBAR_LENGTHS_M:
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
    if epsg is not None and 32601 <= epsg <= 32660:
        zone = epsg - 32600
        return (f"UTM {zone}N (EPSG:{epsg})", "E [m]", "N [m]")
    if epsg is not None and 32701 <= epsg <= 32760:
        zone = epsg - 32700
        return (f"UTM {zone}S (EPSG:{epsg})", "E [m]", "N [m]")
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
    is_sar_fc = False
    if isinstance(raster, xr.Dataset):
        req_bands = list(bands) if bands else list(raster.data_vars)[:3]
        if len(req_bands) == 2 and set(req_bands) == {"VV", "VH"}:
            is_sar_fc = True
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
        if arr3.shape[0] != 3 and arr3.shape[-1] == 3:
            ch_arrays = [arr3[..., 0], arr3[..., 1], arr3[..., 2]]
        else:
            ch_arrays = [arr3[0], arr3[1], arr3[2]]
        legend_bands = list(bands[:3]) if (bands and len(bands) >= 3) else ["R", "G", "B"]
    else:
        raise TypeError("RGB mode requires an xarray.Dataset or 3D xarray.DataArray.")

    valid = np.isfinite(ch_arrays[0]) & np.isfinite(ch_arrays[1]) & np.isfinite(ch_arrays[2])
    if not is_sar_fc:
        valid &= ~((ch_arrays[0] <= 0.0) & (ch_arrays[1] <= 0.0) & (ch_arrays[2] <= 0.0))

    h, w = ch_arrays[0].shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for idx in range(3):
        f_rng = (
            fixed_ranges[idx]
            if (fixed_ranges is not None and idx < len(fixed_ranges))
            else (0.0, 0.35)
        )
        ch_valid = np.where(valid, ch_arrays[idx], np.nan)
        ch01 = _stretch_channel_01(
            ch_valid,
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


def _extent_to_wgs84_bbox(
    extent: tuple[float, float, float, float],
    src_crs_spec: int | str | CRS,
) -> tuple[float, float, float, float]:
    """Convert metric extent ``(min_e, max_e, min_n, max_n)`` to WGS84 ``(min_lon, min_lat, max_lon, max_lat)``."""
    min_e, max_e, min_n, max_n = extent
    src_crs = normalize_crs(src_crs_spec)
    tf = Transformer.from_crs(src_crs, CRS.from_user_input(CRS_WGS84), always_xy=True)
    xs = [min_e, max_e, max_e, min_e]
    ys = [min_n, min_n, max_n, max_n]
    lons, lats = tf.transform(xs, ys)
    return (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats)))


# ---------------------------------------------------------------------------
# Cartographic Drawing Helpers (AOI, Grid, Markers, North, Scale, Inset, Legend)
# ---------------------------------------------------------------------------
def _draw_aoi_outline(ax: plt.Axes, geom_metric: Any, outline_color: str = COLOR_TEAL) -> None:
    """Plot a high-contrast cartographic AOI boundary (contrasting halo + dashed line)."""
    if geom_metric is None or geom_metric.is_empty:
        return

    halo = "#0F172A" if str(outline_color).upper() in ("#FFFFFF", "#FFF", "WHITE") else "#FFFFFF"
    geoms = list(geom_metric.geoms) if hasattr(geom_metric, "geoms") else [geom_metric]
    for poly in geoms:
        if hasattr(poly, "exterior"):
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color=halo, linewidth=2.8, alpha=0.88, zorder=7)
            ax.plot(xs, ys, color=outline_color, linewidth=1.65, linestyle="--", zorder=8)
            for interior in poly.interiors:
                ixs, iys = interior.xy
                ax.plot(ixs, iys, color=halo, linewidth=2.2, alpha=0.85, zorder=7)
                ax.plot(ixs, iys, color=outline_color, linewidth=1.25, linestyle="--", zorder=8)


def _draw_custom_markers(
    ax: plt.Axes,
    markers: Sequence[dict[str, Any]],
    crs_spec: int | str | CRS,
    theme_sty: dict[str, Any],
) -> None:
    """Render optional point markers (GNSS stations, epicenters, profile endpoints) with halo labels."""
    if not markers:
        return
    tf_wgs = None
    for m in markers:
        if not isinstance(m, dict):
            continue
        e_val = m.get("e", m.get("x"))
        n_val = m.get("n", m.get("y"))
        if (e_val is None or n_val is None) and ("lat" in m and "lon" in m):
            if tf_wgs is None:
                tf_wgs = Transformer.from_crs(
                    CRS.from_user_input(CRS_WGS84), normalize_crs(crs_spec), always_xy=True
                )
            e_val, n_val = tf_wgs.transform(float(m["lon"]), float(m["lat"]))
        if e_val is None or n_val is None:
            continue
        ev, nv = float(e_val), float(n_val)
        m_col = str(m.get("color") or "#DC2626")
        name = _clean_print_text(str(m.get("name") or ""))
        ax.scatter(
            [ev],
            [nv],
            s=58,
            c=m_col,
            edgecolors="#FFFFFF",
            linewidths=1.4,
            marker="D",
            zorder=14,
        )
        if name:
            ax.annotate(
                name,
                xy=(ev, nv),
                xytext=(7, 6),
                textcoords="offset points",
                fontsize=7.6,
                fontweight="bold",
                color=theme_sty["text_primary"],
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor=theme_sty["overlay_bg"],
                    edgecolor=m_col,
                    linewidth=0.85,
                    alpha=0.92,
                ),
                zorder=15,
            )


def _draw_scale_bar(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    scalebar_m: int,
    scale_denom: int,
    *,
    position: str = "bottom_left",
    style: str = "alternating_bar",
    show_numeric_scale: bool = True,
    theme_sty: dict[str, Any] | None = None,
    corner_offsets: dict[str, float] | None = None,
) -> None:
    """Draw a geodetic scale bar + optional numerical scale in any of the 4 map corners."""
    sty = theme_sty or THEME_STYLES["classic_light"]
    min_e, max_e, min_n, max_n = extent
    width_m = max_e - min_e
    height_m = max_n - min_n

    bar_frac_w = min(max(float(scalebar_m) / max(width_m, 1.0), 0.08), 0.38)
    pad_x_f = 0.022
    pad_y_bot_f = 0.036
    bar_h_f = 0.017 if style == "alternating_bar" else 0.010
    pad_y_top_f = 0.046 if show_numeric_scale else 0.022

    box_w_f = bar_frac_w + 2.0 * pad_x_f
    box_h_f = pad_y_bot_f + bar_h_f + pad_y_top_f
    margin_x_f = 0.024
    margin_y_f = 0.026
    y_stack = float((corner_offsets or {}).get(position, 0.0))

    if position.endswith("left"):
        bx_f = margin_x_f
    else:
        bx_f = 1.0 - margin_x_f - box_w_f

    if position.startswith("bottom"):
        by_f = margin_y_f + y_stack
    else:
        by_f = 1.0 - margin_y_f - box_h_f - y_stack

    if corner_offsets is not None:
        corner_offsets[position] = y_stack + box_h_f + 0.018

    # Convert box & bar to data coordinates so bar length is EXACTLY ``scalebar_m``!
    box_x = min_e + bx_f * width_m
    box_y = min_n + by_f * height_m
    box_w = box_w_f * width_m
    box_h = box_h_f * height_m

    x0 = box_x + pad_x_f * width_m
    y0 = box_y + pad_y_bot_f * height_m
    bar_h = bar_h_f * height_m

    bg = mpatches.FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        boxstyle=f"round,pad=0,rounding_size={min(width_m, height_m) * 0.008}",
        facecolor=sty["overlay_bg"],
        edgecolor=sty["overlay_edge"],
        linewidth=0.75,
        alpha=sty["overlay_alpha"],
        zorder=10,
    )
    ax.add_patch(bg)

    fg_col = sty["text_primary"]
    alt_col = sty["overlay_bg"]

    if style == "simple_line":
        # Clean horizontal line with end + midpoint vertical ticks
        y_line = y0 + 0.35 * bar_h
        tick_h = height_m * 0.012
        ax.plot([x0, x0 + scalebar_m], [y_line, y_line], color=fg_col, linewidth=1.8, zorder=11)
        for frac in (0.0, 0.5, 1.0):
            tx = x0 + frac * scalebar_m
            ax.plot([tx, tx], [y_line, y_line + tick_h], color=fg_col, linewidth=1.4, zorder=11)
    else:
        # 4 classic alternating geodetic segments
        n_seg = 4
        seg_w = float(scalebar_m) / float(n_seg)
        for i in range(n_seg):
            seg_rect = mpatches.Rectangle(
                (x0 + i * seg_w, y0),
                seg_w,
                bar_h,
                facecolor=fg_col if (i % 2 == 0) else alt_col,
                edgecolor=fg_col,
                linewidth=0.75,
                zorder=11,
            )
            ax.add_patch(seg_rect)

    # Tick labels (0, half, full)
    label_y = y0 - height_m * 0.007
    for frac, lbl_txt in (
        (0.0, "0"),
        (0.5, _format_scalebar_label(0.5 * scalebar_m)),
        (1.0, _format_scalebar_label(scalebar_m)),
    ):
        ax.text(
            x0 + frac * scalebar_m,
            label_y,
            lbl_txt,
            fontsize=7.1,
            fontweight="bold",
            color=fg_col,
            ha="center",
            va="top",
            zorder=12,
        )

    if show_numeric_scale:
        ax.text(
            x0 + 0.5 * scalebar_m,
            y0 + bar_h + height_m * 0.011,
            f"M ≈ 1 : {format_metric_number(scale_denom)}",
            fontsize=7.7,
            fontweight="bold",
            color=fg_col,
            ha="center",
            va="bottom",
            zorder=12,
        )


def _draw_north_arrow(
    ax: plt.Axes,
    lang: str = "sl",
    *,
    position: str = "top_right",
    style: str = "classic_compass",
    size: str = "medium",
    theme_sty: dict[str, Any] | None = None,
    corner_offsets: dict[str, float] | None = None,
) -> None:
    """Draw a crisp cartographic North arrow in any corner (3 styles × 3 sizes)."""
    sty = theme_sty or THEME_STYLES["classic_light"]
    trans = ax.transAxes
    fig = ax.figure
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_aspect = max(bbox.width / max(bbox.height, 1.0), 0.35)

    size_h = {"small": 0.098, "medium": 0.132, "large": 0.168}.get(size, 0.132)
    box_h = size_h
    width_ratio = 0.96 if style == "geodetic_star" else 0.78
    box_w = box_h / ax_aspect * width_ratio

    margin_y = 0.025
    margin_x = margin_y / ax_aspect
    y_stack = float((corner_offsets or {}).get(position, 0.0))

    bx = margin_x if position.endswith("left") else (1.0 - margin_x - box_w)
    by = (margin_y + y_stack) if position.startswith("bottom") else (1.0 - margin_y - box_h - y_stack)

    if corner_offsets is not None:
        corner_offsets[position] = y_stack + box_h + 0.018

    frame = mpatches.FancyBboxPatch(
        (bx, by),
        box_w,
        box_h,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        transform=trans,
        facecolor=sty["overlay_bg"],
        edgecolor=sty["overlay_edge"],
        linewidth=0.75,
        alpha=sty["overlay_alpha"],
        zorder=10,
    )
    ax.add_patch(frame)

    fg_col = sty["text_primary"]
    bg_col = sty["overlay_bg"]
    north_letter = "S" if lang == "sl" else "N"
    cx = bx + 0.5 * box_w
    font_sz = {"small": 8.0, "medium": 9.4, "large": 10.8}.get(size, 9.4)

    if style == "minimal_arrow":
        tip_y = by + box_h * 0.72
        base_y = by + box_h * 0.15
        wing_dx = box_w * 0.22
        wing_y = by + box_h * 0.50
        ax.plot([cx, cx], [base_y, tip_y], transform=trans, color=fg_col, linewidth=1.7, zorder=11)
        ax.plot(
            [cx - wing_dx, cx, cx + wing_dx],
            [wing_y, tip_y, wing_y],
            transform=trans,
            color=fg_col,
            linewidth=1.7,
            zorder=11,
        )
        ax.text(
            cx,
            by + box_h * 0.87,
            north_letter,
            transform=trans,
            fontsize=font_sz,
            fontweight="bold",
            color=fg_col,
            ha="center",
            va="center",
            zorder=12,
        )
    elif style == "geodetic_star":
        cy = by + box_h * 0.45
        r_y = box_h * 0.30
        r_x = box_w * 0.30
        in_y = r_y * 0.26
        in_x = r_x * 0.26
        # 4-point compass star (8 triangles, alternating dark/light)
        facets = [
            ([(cx, cy), (cx, cy + r_y), (cx - in_x, cy + in_y)], fg_col),
            ([(cx, cy), (cx, cy + r_y), (cx + in_x, cy + in_y)], bg_col),
            ([(cx, cy), (cx + r_x, cy), (cx + in_x, cy + in_y)], fg_col),
            ([(cx, cy), (cx + r_x, cy), (cx + in_x, cy - in_y)], bg_col),
            ([(cx, cy), (cx, cy - r_y), (cx + in_x, cy - in_y)], fg_col),
            ([(cx, cy), (cx, cy - r_y), (cx - in_x, cy - in_y)], bg_col),
            ([(cx, cy), (cx - r_x, cy), (cx - in_x, cy - in_y)], fg_col),
            ([(cx, cy), (cx - r_x, cy), (cx - in_x, cy + in_y)], bg_col),
        ]
        for pts, fcol in facets:
            ax.add_patch(
                mpatches.Polygon(
                    pts,
                    closed=True,
                    transform=trans,
                    facecolor=fcol,
                    edgecolor=fg_col,
                    linewidth=0.55,
                    zorder=11,
                )
            )
        ax.text(
            cx,
            by + box_h * 0.88,
            north_letter,
            transform=trans,
            fontsize=font_sz,
            fontweight="bold",
            color=sty["accent"],
            ha="center",
            va="center",
            zorder=12,
        )
        e_lbl = "V" if lang == "sl" else "E"
        w_lbl = "Z" if lang == "sl" else "W"
        s_lbl = "J" if lang == "sl" else "S"
        small_fs = max(5.6, font_sz - 3.2)
        ax.text(cx + r_x + box_w * 0.11, cy, e_lbl, transform=trans, fontsize=small_fs, color=fg_col, ha="center", va="center", zorder=12)
        ax.text(cx - r_x - box_w * 0.11, cy, w_lbl, transform=trans, fontsize=small_fs, color=fg_col, ha="center", va="center", zorder=12)
        ax.text(cx, by + box_h * 0.08, s_lbl, transform=trans, fontsize=small_fs, color=fg_col, ha="center", va="center", zorder=12)
    else:
        # classic_compass
        tip_y = by + box_h * 0.71
        base_y = by + box_h * 0.14
        notch_y = by + box_h * 0.27
        wing_dx = box_w * 0.27
        left_poly = mpatches.Polygon(
            [(cx, tip_y), (cx - wing_dx, base_y), (cx, notch_y)],
            closed=True,
            transform=trans,
            facecolor=fg_col,
            edgecolor=fg_col,
            linewidth=0.75,
            zorder=11,
        )
        right_poly = mpatches.Polygon(
            [(cx, tip_y), (cx + wing_dx, base_y), (cx, notch_y)],
            closed=True,
            transform=trans,
            facecolor=bg_col,
            edgecolor=fg_col,
            linewidth=0.75,
            zorder=11,
        )
        ax.add_patch(left_poly)
        ax.add_patch(right_poly)
        ax.text(
            cx,
            by + box_h * 0.86,
            north_letter,
            transform=trans,
            fontsize=font_sz,
            fontweight="bold",
            color=fg_col,
            ha="center",
            va="center",
            zorder=12,
        )


def _draw_inset_locator_map(
    fig: plt.Figure,
    rect_fig: list[float],
    wgs84_bbox: tuple[float, float, float, float],
    theme_sty: dict[str, Any],
    lang: str = "sl",
) -> None:
    """Draw a crisp zero-dependency vector overview locator map (Slovenia outline or global ±3° grid)."""
    min_lon, min_lat, max_lon, max_lat = wgs84_bbox
    c_lon = 0.5 * (min_lon + max_lon)
    c_lat = 0.5 * (min_lat + max_lat)
    in_slo = is_in_slovenia(c_lat, c_lon)

    ix, iy, iw, ih = rect_fig
    # Outer container card
    card = mpatches.FancyBboxPatch(
        (ix, iy),
        iw,
        ih,
        boxstyle="round,pad=0.0,rounding_size=0.005",
        transform=fig.transFigure,
        facecolor=theme_sty["overlay_bg"],
        edgecolor=theme_sty["overlay_edge"],
        linewidth=0.85,
        alpha=0.95,
        zorder=12,
    )
    fig.patches.append(card)

    title_txt = (
        ("Pregledna karta (SLO)" if lang == "sl" else "Locator Map (Slovenia)")
        if in_slo
        else ("Pregledna lokacija" if lang == "sl" else "Overview Locator")
    )
    fig.text(
        ix + 0.008,
        iy + ih - 0.008,
        title_txt,
        fontsize=6.9,
        fontweight="bold",
        color=theme_sty["accent"],
        ha="left",
        va="top",
        zorder=14,
    )

    pad_l = 0.008
    pad_r = 0.008
    pad_b = 0.010 if in_slo else 0.022
    pad_t = 0.024
    in_ax = fig.add_axes(
        [ix + pad_l, iy + pad_b, max(iw - pad_l - pad_r, 0.02), max(ih - pad_b - pad_t, 0.02)]
    )
    in_ax.set_zorder(13)
    in_ax.set_facecolor(theme_sty["inset_bg"])

    if in_slo:
        sl_lons = [p[0] for p in SLOVENIA_OUTLINE_WGS84]
        sl_lats = [p[1] for p in SLOVENIA_OUTLINE_WGS84]
        in_ax.fill(
            sl_lons,
            sl_lats,
            facecolor=theme_sty["inset_land"],
            edgecolor=theme_sty["inset_land_edge"],
            linewidth=0.95,
            zorder=2,
        )
        # Subtle reference cities
        for cname, clon, clat in SLOVENIA_REF_CITIES:
            in_ax.scatter([clon], [clat], s=8, c=theme_sty["text_muted"], zorder=3)
            in_ax.text(
                clon + 0.06,
                clat - 0.02,
                cname,
                fontsize=5.4,
                color=theme_sty["text_muted"],
                ha="left",
                va="center",
                zorder=4,
            )
        # AOI marker + box
        bw = max(max_lon - min_lon, 0.10)
        bh = max(max_lat - min_lat, 0.07)
        in_ax.add_patch(
            mpatches.Rectangle(
                (c_lon - 0.5 * bw, c_lat - 0.5 * bh),
                bw,
                bh,
                facecolor=(0.86, 0.15, 0.15, 0.25),
                edgecolor="#DC2626",
                linewidth=1.3,
                zorder=6,
            )
        )
        in_ax.scatter([c_lon], [c_lat], s=26, c="#DC2626", edgecolors="#FFFFFF", linewidths=0.7, zorder=7)
        in_ax.set_xlim(13.22, 16.78)
        in_ax.set_ylim(45.30, 47.02)
        in_ax.set_aspect(1.0 / math.cos(math.radians(46.15)), adjustable="box")
        in_ax.set_xticks([])
        in_ax.set_yticks([])
    else:
        span = 3.0
        in_ax.set_xlim(c_lon - span, c_lon + span)
        in_ax.set_ylim(c_lat - span * 0.72, c_lat + span * 0.72)
        in_ax.grid(True, color=theme_sty["divider"], linestyle=":", linewidth=0.55, zorder=1)
        bw = max(max_lon - min_lon, 0.28)
        bh = max(max_lat - min_lat, 0.22)
        in_ax.add_patch(
            mpatches.Rectangle(
                (c_lon - 0.5 * bw, c_lat - 0.5 * bh),
                bw,
                bh,
                facecolor=(0.86, 0.15, 0.15, 0.25),
                edgecolor="#DC2626",
                linewidth=1.3,
                zorder=5,
            )
        )
        in_ax.scatter([c_lon], [c_lat], s=24, c="#DC2626", edgecolors="#FFFFFF", linewidths=0.7, zorder=6)
        in_ax.set_xticks([round(c_lon - 2.0, 1), round(c_lon, 1), round(c_lon + 2.0, 1)])
        in_ax.set_yticks([round(c_lat - 1.5, 1), round(c_lat + 1.5, 1)])
        in_ax.tick_params(labelsize=5.2, colors=theme_sty["text_muted"], length=2.0, pad=1.5)
        fig.text(
            ix + iw - 0.008,
            iy + 0.005,
            f"φ={c_lat:.2f}°, λ={c_lon:.2f}°",
            fontsize=5.6,
            color=theme_sty["text_muted"],
            ha="right",
            va="bottom",
            zorder=14,
        )

    for sp in in_ax.spines.values():
        sp.set_edgecolor(theme_sty["divider"])
        sp.set_linewidth(0.6)


def _draw_legend_block(
    fig: plt.Figure,
    rect_fig: list[float],
    *,
    is_rgb: bool,
    legend_bands: list[str],
    discrete_items: Sequence[tuple[int, str, str]] | None,
    mappable: Any,
    vmin: float,
    vmax: float,
    legend_title: str,
    legend_units: str,
    aoi_drawn: bool,
    aoi_outline_color: str,
    aoi_legend_label: str,
    stats_summary: Sequence[tuple[str, str]] | None,
    show_metadata: bool,
    horizontal: bool,
    compact_inside: bool,
    theme_sty: dict[str, Any],
    lang: str,
) -> None:
    """Render the legend (vertical side panel, horizontal bottom panel, or compact inside-map card)."""
    lx, ly, lw, lh = rect_fig
    leg_frame = mpatches.FancyBboxPatch(
        (lx, ly),
        lw,
        lh,
        boxstyle="round,pad=0.0,rounding_size=0.005",
        transform=fig.transFigure,
        facecolor=theme_sty["overlay_bg"] if compact_inside else theme_sty["panel_bg"],
        edgecolor=theme_sty["overlay_edge"] if compact_inside else theme_sty["divider"],
        linewidth=0.85,
        alpha=theme_sty["overlay_alpha"] if compact_inside else 1.0,
        zorder=12 if compact_inside else 2,
    )
    fig.patches.append(leg_frame)
    z_base = 13 if compact_inside else 4

    t_pri = theme_sty["text_primary"]
    t_mut = theme_sty["text_muted"]
    accent = theme_sty["accent"]
    div_col = theme_sty["divider"]

    leg_heading = "LEGENDA" if lang == "sl" else "LEGEND"
    head_fs = 7.4 if compact_inside else 8.5
    body_fs = 6.8 if compact_inside else 7.7
    fig.text(
        lx + 0.010,
        ly + lh - (0.010 if compact_inside else 0.016),
        leg_heading,
        fontsize=head_fs,
        fontweight="bold",
        color=accent,
        ha="left",
        va="top",
        zorder=z_base,
    )
    div_y = ly + lh - (0.028 if compact_inside else 0.036)
    fig.lines.append(
        plt.Line2D(
            [lx + 0.008, lx + lw - 0.008],
            [div_y, div_y],
            transform=fig.transFigure,
            color=div_col,
            linewidth=0.65,
            zorder=z_base,
        )
    )

    if is_rgb:
        subtitle_lbl = legend_title or (
            "Kombinacija pasov" if lang == "sl" else "Band Combination"
        )
        fig.text(
            lx + 0.010,
            div_y - 0.008,
            subtitle_lbl,
            fontsize=body_fs + 0.3,
            fontweight="bold",
            color=t_pri,
            ha="left",
            va="top",
            zorder=z_base,
        )
        ch_colors = ("#DC2626", "#16A34A", "#2563EB")
        ch_prefixes = ("R", "G", "B")
        if not horizontal:
            step_y = 0.028 if compact_inside else 0.035
            cur_y = div_y - (0.034 if compact_inside else 0.044)
            for idx, b_code in enumerate(legend_bands[:3]):
                b_desc = BAND_WAVELENGTH_LABELS.get(b_code, b_code)
                if compact_inside:
                    b_desc = b_code
                fig.patches.append(
                    mpatches.Rectangle(
                        (lx + 0.011, cur_y - 0.009),
                        0.013,
                        0.015,
                        transform=fig.transFigure,
                        facecolor=ch_colors[idx],
                        edgecolor=theme_sty["border"],
                        linewidth=0.5,
                        zorder=z_base,
                    )
                )
                fig.text(
                    lx + 0.029,
                    cur_y - 0.001,
                    f"{ch_prefixes[idx]}: {b_desc}",
                    fontsize=body_fs,
                    color=t_pri,
                    ha="left",
                    va="center",
                    zorder=z_base,
                )
                cur_y -= step_y
            if aoi_drawn and cur_y > ly + 0.018:
                cur_y -= 0.006
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.011, lx + 0.028],
                        [cur_y, cur_y],
                        transform=fig.transFigure,
                        color=aoi_outline_color,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=z_base,
                    )
                )
                fig.text(
                    lx + 0.033,
                    cur_y,
                    aoi_legend_label,
                    fontsize=body_fs,
                    color=t_pri,
                    ha="left",
                    va="center",
                    zorder=z_base,
                )
                cur_y -= step_y
            if show_metadata and stats_summary and not compact_inside and cur_y > ly + 0.025:
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.008, lx + lw - 0.008],
                        [cur_y + 0.012, cur_y + 0.012],
                        transform=fig.transFigure,
                        color=div_col,
                        linewidth=0.6,
                        zorder=z_base,
                    )
                )
                for k_lbl, v_lbl in stats_summary[:5]:
                    if cur_y < ly + 0.012:
                        break
                    fig.text(lx + 0.011, cur_y, f"{k_lbl}:", fontsize=7.2, color=t_mut, ha="left", va="center", zorder=z_base)
                    fig.text(lx + lw - 0.010, cur_y, v_lbl, fontsize=7.3, fontweight="bold", color=t_pri, ha="right", va="center", zorder=z_base)
                    cur_y -= 0.026
        else:
            step_x = (lw - 0.028) / 3.0
            row_y = ly + lh * 0.36
            for idx, b_code in enumerate(legend_bands[:3]):
                bx_i = lx + 0.014 + idx * step_x
                b_desc = BAND_WAVELENGTH_LABELS.get(b_code, b_code)
                fig.patches.append(
                    mpatches.Rectangle(
                        (bx_i, row_y - 0.008),
                        0.016,
                        0.015,
                        transform=fig.transFigure,
                        facecolor=ch_colors[idx],
                        edgecolor=theme_sty["border"],
                        linewidth=0.5,
                        zorder=z_base,
                    )
                )
                fig.text(
                    bx_i + 0.022,
                    row_y,
                    f"{ch_prefixes[idx]}: {b_desc}",
                    fontsize=7.4,
                    color=t_pri,
                    ha="left",
                    va="center",
                    zorder=z_base,
                )
            if aoi_drawn:
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.014, lx + 0.038],
                        [ly + lh * 0.14, ly + lh * 0.14],
                        transform=fig.transFigure,
                        color=aoi_outline_color,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=z_base,
                    )
                )
                fig.text(
                    lx + 0.045,
                    ly + lh * 0.14,
                    aoi_legend_label,
                    fontsize=7.3,
                    color=t_pri,
                    ha="left",
                    va="center",
                    zorder=z_base,
                )

    elif discrete_items:
        title_str = legend_title or ("Razredi" if lang == "sl" else "Classes")
        fig.text(
            lx + 0.010,
            div_y - 0.008,
            title_str,
            fontsize=body_fs + 0.3,
            fontweight="bold",
            color=t_pri,
            ha="left",
            va="top",
            zorder=z_base,
        )
        if not horizontal:
            step_y = 0.026 if compact_inside else 0.033
            cur_y = div_y - (0.034 if compact_inside else 0.044)
            for _val, cls_lbl, hex_col in discrete_items:
                if cur_y < ly + 0.014:
                    break
                fig.patches.append(
                    mpatches.Rectangle(
                        (lx + 0.011, cur_y - 0.009),
                        0.014,
                        0.015,
                        transform=fig.transFigure,
                        facecolor=hex_col,
                        edgecolor=theme_sty["border"],
                        linewidth=0.5,
                        zorder=z_base,
                    )
                )
                fig.text(
                    lx + 0.030,
                    cur_y - 0.001,
                    cls_lbl,
                    fontsize=body_fs,
                    color=t_pri,
                    ha="left",
                    va="center",
                    zorder=z_base,
                )
                cur_y -= step_y
            if aoi_drawn and cur_y > ly + 0.018:
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.011, lx + 0.028],
                        [cur_y, cur_y],
                        transform=fig.transFigure,
                        color=aoi_outline_color,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=z_base,
                    )
                )
                fig.text(lx + 0.033, cur_y, aoi_legend_label, fontsize=body_fs, color=t_pri, ha="left", va="center", zorder=z_base)
                cur_y -= step_y
            if show_metadata and stats_summary and not compact_inside and cur_y > ly + 0.025:
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.008, lx + lw - 0.008],
                        [cur_y + 0.012, cur_y + 0.012],
                        transform=fig.transFigure,
                        color=div_col,
                        linewidth=0.6,
                        zorder=z_base,
                    )
                )
                for k_lbl, v_lbl in stats_summary[:5]:
                    if cur_y < ly + 0.012:
                        break
                    fig.text(lx + 0.011, cur_y, f"{k_lbl}:", fontsize=7.2, color=t_mut, ha="left", va="center", zorder=z_base)
                    fig.text(lx + lw - 0.010, cur_y, v_lbl, fontsize=7.3, fontweight="bold", color=t_pri, ha="right", va="center", zorder=z_base)
                    cur_y -= 0.026
        else:
            n_cls = max(len(discrete_items), 1)
            step_x = (lw - 0.028) / float(min(n_cls, 4))
            row_y = ly + lh * 0.36
            for idx, (_val, cls_lbl, hex_col) in enumerate(discrete_items[:4]):
                bx_i = lx + 0.014 + idx * step_x
                fig.patches.append(
                    mpatches.Rectangle(
                        (bx_i, row_y - 0.008),
                        0.016,
                        0.015,
                        transform=fig.transFigure,
                        facecolor=hex_col,
                        edgecolor=theme_sty["border"],
                        linewidth=0.5,
                        zorder=z_base,
                    )
                )
                fig.text(bx_i + 0.022, row_y, cls_lbl, fontsize=7.4, color=t_pri, ha="left", va="center", zorder=z_base)

    elif mappable is not None:
        unit_suffix = f" [{legend_units}]" if legend_units else ""
        cbar_title = f"{legend_title}{unit_suffix}" if legend_title else (
            "Vrednost" if lang == "sl" else "Value"
        )
        fig.text(
            lx + 0.010,
            div_y - 0.008,
            cbar_title,
            fontsize=body_fs + 0.3,
            fontweight="bold",
            color=t_pri,
            ha="left",
            va="top",
            zorder=z_base,
        )
        ticks = np.linspace(float(vmin), float(vmax), 5)
        tick_fmt = "{:.1f}" if abs(vmax - vmin) >= 5.0 else "{:.2f}"

        if horizontal or compact_inside:
            cax_y = ly + lh * (0.28 if horizontal else 0.25)
            cax_h = lh * (0.22 if horizontal else 0.20)
            cax = fig.add_axes([lx + 0.018, cax_y, max(lw - 0.036, 0.04), max(cax_h, 0.012)])
            cax.set_zorder(z_base + 1)
            cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            cb.set_ticks(ticks if horizontal else [ticks[0], ticks[2], ticks[-1]])
            cb.set_ticklabels([tick_fmt.format(tv) for tv in (ticks if horizontal else [ticks[0], ticks[2], ticks[-1]])])
            cb.ax.tick_params(labelsize=6.6 if compact_inside else 7.3, colors=t_pri, length=2.5)
            cb.outline.set_edgecolor(theme_sty["border"])
            cb.outline.set_linewidth(0.6)
        else:
            cbar_h = min(lh * 0.42, 0.29)
            cbar_y = div_y - 0.040 - cbar_h
            cax = fig.add_axes([lx + 0.020, cbar_y, 0.022, cbar_h])
            cax.set_zorder(z_base + 1)
            cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
            cb.set_ticks(ticks)
            cb.set_ticklabels([tick_fmt.format(tv) for tv in ticks])
            cb.ax.tick_params(labelsize=7.4, colors=t_pri, length=3.0)
            cb.outline.set_edgecolor(theme_sty["border"])
            cb.outline.set_linewidth(0.65)

            cur_y = cbar_y - 0.032
            if aoi_drawn and cur_y > ly + 0.020:
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.011, lx + 0.028],
                        [cur_y, cur_y],
                        transform=fig.transFigure,
                        color=aoi_outline_color,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=z_base,
                    )
                )
                fig.text(lx + 0.033, cur_y, aoi_legend_label, fontsize=7.5, color=t_pri, ha="left", va="center", zorder=z_base)
                cur_y -= 0.030
            if show_metadata and stats_summary and cur_y > ly + 0.025:
                fig.lines.append(
                    plt.Line2D(
                        [lx + 0.008, lx + lw - 0.008],
                        [cur_y + 0.012, cur_y + 0.012],
                        transform=fig.transFigure,
                        color=div_col,
                        linewidth=0.6,
                        zorder=z_base,
                    )
                )
                for k_lbl, v_lbl in stats_summary[:5]:
                    if cur_y < ly + 0.012:
                        break
                    fig.text(lx + 0.011, cur_y, f"{k_lbl}:", fontsize=7.2, color=t_mut, ha="left", va="center", zorder=z_base)
                    fig.text(lx + lw - 0.010, cur_y, v_lbl, fontsize=7.3, fontweight="bold", color=t_pri, ha="right", va="center", zorder=z_base)
                    cur_y -= 0.025


# ---------------------------------------------------------------------------
# Main QGIS-Style Geodetic Print Layout Renderer
# ---------------------------------------------------------------------------
def render_geodetic_map(
    raster: xr.Dataset | xr.DataArray | dict[str, Any],
    crs: int | str | CRS = CRS_D96TM,
    *,
    layout_config: dict[str, Any] | None = None,
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
    dpi: int | None = None,
    preview_dpi: int = 150,
    export_dpi: int = 300,
) -> dict[str, Any]:
    """Render a publication-grade geodetic map in strict metric projection (PNG + PDF).

    Accepts EITHER:
    - A ``layer_info`` dict (from Workspace 4 / AI Assistant) + optional ``layout_config`` dict, OR
    - Direct ``raster`` (``xr.Dataset | xr.DataArray``) + keyword args and/or ``layout_config`` dict
      (100% backward compatible with all legacy calls).
    """
    # Unpack layer_info dict when passed as first argument
    if isinstance(raster, dict):
        layer_info = raster
        raster = layer_info["raster"]
        crs = layer_info.get("crs", crs)
        is_rgb = bool(layer_info.get("is_rgb", is_rgb))
        bands = layer_info.get("bands", bands)
        use_percentile_stretch = bool(layer_info.get("use_percentile_stretch", use_percentile_stretch))
        gamma = float(layer_info.get("gamma", gamma))
        fixed_ranges = layer_info.get("fixed_ranges", fixed_ranges)
        cmap_name = str(layer_info.get("cmap_name", cmap_name))
        vmin = float(layer_info.get("vmin", vmin))
        vmax = float(layer_info.get("vmax", vmax))
        discrete_items = layer_info.get("discrete_items", discrete_items)
        legend_title = str(layer_info.get("legend_title", legend_title))
        legend_units = str(layer_info.get("legend_units", legend_units))
        stats_summary = layer_info.get("stats_summary", stats_summary)
        sensor_label = str(layer_info.get("sensor_label", sensor_label))
        date_label = str(layer_info.get("date_label", date_label))
        resolution_m = float(layer_info.get("resolution_m", resolution_m))
        if aoi_polygon_geojson is None:
            aoi_polygon_geojson = layer_info.get("aoi_polygon_4326", layer_info.get("aoi_polygon_geojson"))
        if aoi_bbox_4326 is None:
            aoi_bbox_4326 = layer_info.get("aoi_bbox_4326")
        if not title or title == "Satelitska karta območja":
            title = str(layer_info.get("default_title", title))
        if not subtitle:
            subtitle = str(layer_info.get("default_subtitle", subtitle))

    # Build unified MapLayoutConfig (legacy kwargs provide base, layout_config overrides them cleanly)
    base_overrides: dict[str, Any] = {
        "paper_format": page_format,
        "dpi_preview": int(dpi) if dpi is not None else preview_dpi,
        "dpi_export": export_dpi,
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "legend_title": legend_title,
        "show_grid": show_grid,
        "grid_spacing": "auto" if (grid_spacing_m is None or int(grid_spacing_m) <= 0) else int(grid_spacing_m),
        "grid_color": grid_style if grid_style in _VALID_GRID_COLORS else "light",
        "grid_style": grid_style if grid_style in _VALID_GRID_STYLES else "lines",
        "show_scale_bar": show_scalebar,
        "show_north_arrow": show_north_arrow,
        "show_legend": show_legend,
        "show_aoi_outline": show_aoi_outline,
    }
    if isinstance(layout_config, dict):
        base_overrides.update(layout_config)
    cfg = merge_layout_config(base_overrides)

    # Clean printable strings
    eff_title = _clean_print_text(cfg["title"] or title) or (
        "Satelitska karta območja" if lang == "sl" else "Satellite Map of Area of Interest"
    )
    eff_subtitle = _clean_print_text(cfg["subtitle"])
    eff_author = _clean_print_text(cfg["author"])
    eff_legend_title = _clean_print_text(cfg["legend_title"] or legend_title)
    eff_note = _clean_print_text(cfg["custom_note_text"])
    sensor_label = _clean_print_text(sensor_label)
    if discrete_items:
        discrete_items = [(int(v), _clean_print_text(lbl), str(col)) for v, lbl, col in discrete_items]

    fmt_spec = PAGE_FORMATS[cfg["paper_format"]]
    fig_w_in, fig_h_in = fmt_spec["figsize"]
    orientation = fmt_spec["orientation"]
    sty = THEME_STYLES[cfg["theme"]]

    min_e, max_e, min_n, max_n, y_ascending = _extract_extent_and_orientation(raster)
    extent = (min_e, max_e, min_n, max_n)
    width_m = max(max_e - min_e, 1.0)
    height_m = max(max_n - min_n, 1.0)

    crs_title, x_lbl, y_lbl = _crs_display_info(crs, lang=lang)
    eff_grid_m = (
        int(cfg["grid_spacing"])
        if isinstance(cfg["grid_spacing"], int) and cfg["grid_spacing"] > 0
        else compute_nice_grid_spacing(max(width_m, height_m), target_lines=5)
    )
    scalebar_m = compute_nice_scalebar_length(width_m, target_fraction=0.22)

    aoi_metric = (
        _project_aoi_to_crs(crs, aoi_polygon_geojson=aoi_polygon_geojson, aoi_bbox_4326=aoi_bbox_4326)
        if cfg["show_aoi_outline"]
        else None
    )
    aoi_lbl = aoi_legend_label or ("Meja AOI" if lang == "sl" else "AOI Boundary")

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
    # Build Matplotlib Figure with Collision-Free QGIS-Style Layout Regions
    # -----------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": sty["border"],
            "axes.linewidth": float(cfg["map_border_width"]),
        }
    )

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), facecolor=sty["paper_bg"])
    try:
        # Outer neatline border (omitted in minimal_white)
        if sty["sheet_border"]:
            fig.patches.append(
                mpatches.Rectangle(
                    (0.015, 0.015),
                    0.970,
                    0.970,
                    transform=fig.transFigure,
                    facecolor="none",
                    edgecolor=sty["border"],
                    linewidth=1.1,
                    zorder=10,
                )
            )

        # Header band (when show_title & top_left/top_center)
        title_pos = cfg["title_position"]
        has_outer_header = bool(cfg["show_title"]) and title_pos in ("top_left", "top_center")
        header_top = 0.968
        if has_outer_header:
            header_h = 0.086 if orientation == "landscape" else 0.070
            header_y = header_top - header_h
            if title_pos == "top_left":
                fig.patches.append(
                    mpatches.Rectangle(
                        (0.025, header_y + 0.008),
                        0.006,
                        header_h - 0.014,
                        transform=fig.transFigure,
                        facecolor=sty["accent"],
                        edgecolor="none",
                        zorder=2,
                    )
                )
            fig.lines.append(
                plt.Line2D(
                    [0.025, 0.975],
                    [header_y, header_y],
                    transform=fig.transFigure,
                    color=sty["divider"],
                    linewidth=0.85,
                )
            )
            title_x = 0.50 if title_pos == "top_center" else 0.039
            title_ha = "center" if title_pos == "top_center" else "left"
            fig.text(
                title_x,
                header_y + header_h * (0.64 if eff_subtitle else 0.50),
                eff_title,
                fontsize=14.5 if orientation == "landscape" else 13.2,
                fontweight="bold",
                color=sty["text_primary"],
                ha=title_ha,
                va="center",
            )
            if eff_subtitle:
                fig.text(
                    title_x,
                    header_y + header_h * 0.24,
                    eff_subtitle,
                    fontsize=9.1,
                    color=sty["text_secondary"],
                    ha=title_ha,
                    va="center",
                )
            if eff_author and title_pos == "top_left":
                author_prefix = "Avtor:" if lang == "sl" else "Author:"
                fig.text(
                    0.968,
                    header_y + header_h * 0.62,
                    f"{author_prefix} {eff_author}",
                    fontsize=8.5,
                    fontweight="bold",
                    color=sty["text_secondary"],
                    ha="right",
                    va="center",
                )
            if date_label and title_pos == "top_left":
                fig.text(
                    0.968,
                    header_y + header_h * 0.24,
                    f"{sensor_label}  ·  {date_label}",
                    fontsize=8.1,
                    color=sty["text_muted"],
                    ha="right",
                    va="center",
                )
        else:
            header_y = 0.965

        # Footer band
        has_footer = bool(cfg["show_footer"])
        footer_y = 0.022
        footer_h = (0.066 if orientation == "landscape" else 0.058) if has_footer else 0.0
        footer_top = footer_y + footer_h
        if has_footer:
            fig.patches.append(
                mpatches.Rectangle(
                    (0.025, footer_y),
                    0.950,
                    footer_h,
                    transform=fig.transFigure,
                    facecolor=sty["footer_bg"],
                    edgecolor=sty["divider"],
                    linewidth=0.75,
                    zorder=1,
                )
            )

        # Side / Bottom Panels
        leg_pos = cfg["legend_position"]
        inset_pos = cfg["inset_position"]
        want_right_leg = bool(cfg["show_legend"]) and leg_pos == "right_panel"
        want_bottom_leg = bool(cfg["show_legend"]) and leg_pos == "bottom_panel"
        want_right_inset = bool(cfg["show_inset_map"]) and inset_pos == "right_panel_bottom"
        want_note_below_leg = bool(eff_note) and cfg["custom_note_position"] == "below_legend"

        has_right_panel = want_right_leg or want_right_inset
        right_panel_x = 0.768 if orientation == "landscape" else 0.742
        right_panel_w = 0.972 - right_panel_x

        leg_bot_h = (0.125 if orientation == "portrait" else 0.132) if want_bottom_leg else 0.0
        map_bot = footer_top + (leg_bot_h + 0.068 if want_bottom_leg else 0.058)
        map_top_pad = 0.044
        map_left = 0.086 if orientation == "landscape" else 0.098
        map_right = (right_panel_x - 0.040) if has_right_panel else 0.954
        map_rect = [
            map_left,
            map_bot,
            max(map_right - map_left, 0.30),
            max(header_y - map_bot - map_top_pad, 0.30),
        ]

        # Compute exact panel rectangles for legend & right-panel inset
        panel_bot = footer_top + (leg_bot_h + 0.028 if want_bottom_leg else 0.024)
        panel_top = header_y - 0.020
        panel_h = max(panel_top - panel_bot, 0.20)

        leg_panel_rect: list[float] | None = None
        inset_panel_rect: list[float] | None = None
        note_below_leg_y: float | None = None

        if want_bottom_leg:
            leg_panel_rect = [0.032, footer_top + 0.012, 0.936, leg_bot_h]

        if has_right_panel:
            note_reserve = 0.055 if (want_note_below_leg and want_right_leg) else 0.0
            if want_right_leg and want_right_inset:
                in_h = min(panel_h * 0.34, 0.255)
                inset_panel_rect = [right_panel_x, panel_bot, right_panel_w, in_h]
                leg_y0 = panel_bot + in_h + 0.018 + note_reserve
                leg_panel_rect = [right_panel_x, leg_y0, right_panel_w, max(panel_top - leg_y0, 0.18)]
                if want_note_below_leg:
                    note_below_leg_y = panel_bot + in_h + 0.014
            elif want_right_leg:
                leg_y0 = panel_bot + note_reserve
                leg_panel_rect = [right_panel_x, leg_y0, right_panel_w, max(panel_top - leg_y0, 0.18)]
                if want_note_below_leg:
                    note_below_leg_y = panel_bot + 0.008
            elif want_right_inset:
                in_h = min(panel_h * 0.42, 0.28)
                inset_panel_rect = [right_panel_x, panel_bot, right_panel_w, in_h]

        # -------------------------------------------------------------------
        # Map Axes (Strict 1:1 Metric Aspect Ratio)
        # -------------------------------------------------------------------
        ax = fig.add_axes(map_rect)
        ax.set_zorder(2)
        ax.set_facecolor(sty["nodata_bg"])

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
            cmap.set_bad(color=sty["nodata_bg"])
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

        for sp in ax.spines.values():
            sp.set_edgecolor(sty["border"])
            sp.set_linewidth(float(cfg["map_border_width"]))

        # AOI boundary overlay
        if aoi_metric is not None:
            _draw_aoi_outline(ax, aoi_metric, outline_color=str(cfg["aoi_outline_color"]))

        # Optional custom markers
        if cfg["markers"]:
            _draw_custom_markers(ax, cfg["markers"], crs_spec=crs, theme_sty=sty)

        # Metric coordinate ticks & grid styles (lines | crosses | ticks_only)
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
            labelsize=7.5,
            colors=sty["text_primary"],
            top=True,
            right=True,
            labeltop=True,
            labelright=False,
        )
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(90)
            lbl.set_va("center")

        ax.set_xlabel(x_lbl, fontsize=8.1, fontweight="bold", color=sty["text_primary"], labelpad=4)
        ax.set_ylabel(y_lbl, fontsize=8.1, fontweight="bold", color=sty["text_primary"], labelpad=4)

        if cfg["show_grid"]:
            g_col_map = {
                "light": "#FFFFFF",
                "dark": COLOR_SLATE_900,
                "teal": "#14B8A6",
            }
            g_color = g_col_map.get(cfg["grid_color"], "#FFFFFF")
            g_alpha = float(cfg["grid_alpha"])
            g_mode = cfg["grid_style"]
            if g_mode == "lines":
                ax.grid(
                    True,
                    which="major",
                    color=g_color,
                    linestyle="-",
                    linewidth=0.65,
                    alpha=g_alpha,
                    zorder=5,
                )
            elif g_mode == "crosses":
                ax.grid(False)
                gx, gy = np.meshgrid(x_ticks, y_ticks)
                ax.plot(
                    gx.ravel(),
                    gy.ravel(),
                    marker="+",
                    linestyle="none",
                    markersize=8.5,
                    markeredgewidth=0.95,
                    color=g_color,
                    alpha=min(1.0, g_alpha + 0.22),
                    zorder=6,
                )
            else:
                # ticks_only: outer metric ticks without interior lines
                ax.grid(False)
        else:
            ax.grid(False)

        # Force draw so we get EXACT rendered map box on paper for true scale & inside overlays!
        fig.canvas.draw()
        ax_pos = ax.get_position()
        paper_width_m = float(ax_pos.width * fig_w_in) * 0.0254
        scale_denom = compute_nice_scale_denom(width_m, paper_width_m)

        # Corner collision-stacking tracker (in axes-fraction height reserved per corner)
        corner_offsets: dict[str, float] = {
            "top_left": 0.0,
            "top_right": 0.0,
            "bottom_left": 0.0,
            "bottom_right": 0.0,
        }

        # Inside-map title block (when title_position is inside_top_left / inside_top_center)
        if cfg["show_title"] and title_pos in ("inside_top_left", "inside_top_center"):
            tx = 0.024 if title_pos == "inside_top_left" else 0.50
            tha = "left" if title_pos == "inside_top_left" else "center"
            t_block = f"{eff_title}\n{eff_subtitle}" if eff_subtitle else eff_title
            ax.text(
                tx,
                0.972,
                t_block,
                transform=ax.transAxes,
                fontsize=10.5 if not eff_subtitle else 9.6,
                fontweight="bold",
                color=sty["text_primary"],
                ha=tha,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.38",
                    facecolor=sty["overlay_bg"],
                    edgecolor=sty["accent"],
                    linewidth=1.0,
                    alpha=sty["overlay_alpha"],
                ),
                zorder=15,
            )
            if title_pos == "inside_top_left":
                corner_offsets["top_left"] += 0.135 if eff_subtitle else 0.095

        # North arrow
        if cfg["show_north_arrow"]:
            _draw_north_arrow(
                ax,
                lang=lang,
                position=cfg["north_position"],
                style=cfg["north_style"],
                size=cfg["north_size"],
                theme_sty=sty,
                corner_offsets=corner_offsets,
            )

        # Scale bar
        if cfg["show_scale_bar"]:
            _draw_scale_bar(
                ax,
                extent=extent,
                scalebar_m=scalebar_m,
                scale_denom=scale_denom,
                position=cfg["scale_position"],
                style=cfg["scale_style"],
                show_numeric_scale=bool(cfg["show_numeric_scale"]),
                theme_sty=sty,
                corner_offsets=corner_offsets,
            )

        # Legend (right_panel, bottom_panel, OR inside-map card)
        if cfg["show_legend"]:
            if leg_panel_rect is not None:
                _draw_legend_block(
                    fig,
                    leg_panel_rect,
                    is_rgb=is_rgb,
                    legend_bands=legend_bands,
                    discrete_items=discrete_items,
                    mappable=mappable,
                    vmin=float(vmin),
                    vmax=float(vmax),
                    legend_title=eff_legend_title,
                    legend_units=legend_units,
                    aoi_drawn=(aoi_metric is not None),
                    aoi_outline_color=str(cfg["aoi_outline_color"]),
                    aoi_legend_label=aoi_lbl,
                    stats_summary=stats_summary,
                    show_metadata=bool(cfg["show_metadata_in_legend"]),
                    horizontal=want_bottom_leg,
                    compact_inside=False,
                    theme_sty=sty,
                    lang=lang,
                )
            elif leg_pos in ("inside_bottom_right", "inside_top_right", "inside_bottom_left"):
                corner_k = leg_pos.replace("inside_", "")
                card_w_ax = 0.28 if (not is_rgb and not discrete_items) else 0.25
                card_h_ax = 0.18 if (not is_rgb and not discrete_items) else min(
                    0.30, 0.10 + 0.036 * max(len(discrete_items or legend_bands), 2)
                )
                y_stk = corner_offsets.get(corner_k, 0.0)
                mx, my = 0.024, 0.026
                cx_ax = mx if corner_k.endswith("left") else (1.0 - mx - card_w_ax)
                cy_ax = (my + y_stk) if corner_k.startswith("bottom") else (1.0 - my - card_h_ax - y_stk)
                corner_offsets[corner_k] = y_stk + card_h_ax + 0.018
                inside_leg_rect = [
                    ax_pos.x0 + cx_ax * ax_pos.width,
                    ax_pos.y0 + cy_ax * ax_pos.height,
                    card_w_ax * ax_pos.width,
                    card_h_ax * ax_pos.height,
                ]
                _draw_legend_block(
                    fig,
                    inside_leg_rect,
                    is_rgb=is_rgb,
                    legend_bands=legend_bands,
                    discrete_items=discrete_items,
                    mappable=mappable,
                    vmin=float(vmin),
                    vmax=float(vmax),
                    legend_title=eff_legend_title,
                    legend_units=legend_units,
                    aoi_drawn=(aoi_metric is not None),
                    aoi_outline_color=str(cfg["aoi_outline_color"]),
                    aoi_legend_label=aoi_lbl,
                    stats_summary=None,
                    show_metadata=False,
                    horizontal=False,
                    compact_inside=True,
                    theme_sty=sty,
                    lang=lang,
                )

        # Overview / Inset Locator Map
        if cfg["show_inset_map"]:
            wgs_bbox = (
                tuple(float(v) for v in aoi_bbox_4326)
                if (aoi_bbox_4326 is not None and len(aoi_bbox_4326) == 4)
                else _extent_to_wgs84_bbox(extent, crs)
            )
            if inset_panel_rect is not None:
                _draw_inset_locator_map(fig, inset_panel_rect, wgs_bbox, theme_sty=sty, lang=lang)
            elif inset_pos in _VALID_CORNERS:
                in_w_ax = 0.26
                in_h_ax = 0.24
                y_stk = corner_offsets.get(inset_pos, 0.0)
                mx, my = 0.024, 0.026
                ix_ax = mx if inset_pos.endswith("left") else (1.0 - mx - in_w_ax)
                iy_ax = (my + y_stk) if inset_pos.startswith("bottom") else (1.0 - my - in_h_ax - y_stk)
                corner_offsets[inset_pos] = y_stk + in_h_ax + 0.018
                in_rect_fig = [
                    ax_pos.x0 + ix_ax * ax_pos.width,
                    ax_pos.y0 + iy_ax * ax_pos.height,
                    in_w_ax * ax_pos.width,
                    in_h_ax * ax_pos.height,
                ]
                _draw_inset_locator_map(fig, in_rect_fig, wgs_bbox, theme_sty=sty, lang=lang)

        # Custom Note / Thesis Caption
        note_pos = cfg["custom_note_position"]
        if eff_note and note_pos != "none":
            if note_pos == "below_legend" and note_below_leg_y is not None:
                fig.text(
                    right_panel_x + 0.004,
                    note_below_leg_y + 0.022,
                    eff_note,
                    fontsize=7.0,
                    style="italic",
                    color=sty["text_secondary"],
                    ha="left",
                    va="center",
                    wrap=True,
                    zorder=12,
                )
            elif note_pos in ("bottom_left", "bottom_right"):
                y_stk = corner_offsets.get(note_pos, 0.0)
                nx_ax = 0.024 if note_pos == "bottom_left" else 0.976
                nha = "left" if note_pos == "bottom_left" else "right"
                ax.text(
                    nx_ax,
                    0.026 + y_stk,
                    eff_note,
                    transform=ax.transAxes,
                    fontsize=7.2,
                    style="italic",
                    color=sty["text_primary"],
                    ha=nha,
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.28",
                        facecolor=sty["overlay_bg"],
                        edgecolor=sty["divider"],
                        linewidth=0.65,
                        alpha=sty["overlay_alpha"],
                    ),
                    zorder=14,
                )

        # -------------------------------------------------------------------
        # Geodetic Technical Metadata Footer
        # -------------------------------------------------------------------
        if has_footer:
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

            fig.text(0.036, footer_y + footer_h * 0.68, row1_left, fontsize=7.5, fontweight="bold", color=sty["text_primary"], ha="left", va="center")
            fig.text(0.964, footer_y + footer_h * 0.68, row1_right, fontsize=7.4, color=sty["text_secondary"], ha="right", va="center")
            fig.text(0.036, footer_y + footer_h * 0.28, row2_left, fontsize=7.2, color=sty["text_secondary"], ha="left", va="center")
            fig.text(0.964, footer_y + footer_h * 0.28, row2_right, fontsize=7.0, color=sty["text_muted"], ha="right", va="center")

        # -------------------------------------------------------------------
        # Export Buffers (Preview PNG, 300 DPI Print PNG, 300 DPI Vector PDF)
        # -------------------------------------------------------------------
        buf_prev = io.BytesIO()
        fig.savefig(buf_prev, format="png", dpi=int(cfg["dpi_preview"]), facecolor=sty["paper_bg"])
        png_preview_bytes = buf_prev.getvalue()
        buf_prev.close()

        buf_hi = io.BytesIO()
        fig.savefig(buf_hi, format="png", dpi=int(cfg["dpi_export"]), facecolor=sty["paper_bg"])
        png_hires_bytes = buf_hi.getvalue()
        buf_hi.close()

        buf_pdf = io.BytesIO()
        fig.savefig(
            buf_pdf,
            format="pdf",
            dpi=int(cfg["dpi_export"]),
            facecolor=sty["paper_bg"],
            metadata={
                "Title": eff_title,
                "Author": eff_author,
                "Subject": f"{sensor_label} | {crs_title}",
                "Creator": "Daljinsko zaznavanje — QGIS-Style Geodetic Composer",
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
            "layout_config": cfg,
        }
    finally:
        plt.close(fig)
        gc.collect()
